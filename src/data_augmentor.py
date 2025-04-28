import itertools
import json
import os
import pickle
import random
from collections import namedtuple, defaultdict
from datetime import datetime
from statistics import mean, median, stdev, variance
from typing import NoReturn

import fire
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

from src.metrics.cherrant import cherrant_scores
from src.dataset import Dataset, YACLCDataset
from src.data_structure import Sentence, Annotation, Worker

Alternatives = namedtuple('Alternatives', ('higher_anno', 'lower_anno', 'closest_anno'))


class DataAugmentor:

    def __init__(self, original_dataset: Dataset):
        self.original_dataset: Dataset = original_dataset
        self.augmented_dataset: Dataset

    @staticmethod
    def evaluate_annos(system_annos: list[Annotation], gold_annos: list[Annotation]) -> NoReturn:
        pass

    def augment_sent(self, sent: Sentence):
        pass

    def augment_dataset(self, processes: int = 8):
        pass


class YACLCDataAugmentor(DataAugmentor):

    def __init__(self, original_dataset: Dataset):
        super().__init__(original_dataset)
        self.augmented_dataset: YACLCDataset = YACLCDataset()

        anno2score = self.evaluate_annos(original_dataset.annos,
                                         [original_dataset.sent2gold_anno[anno.sent] for anno in
                                          original_dataset.annos])

        for anno in original_dataset.annos:
            anno.score = anno2score[anno]

    @staticmethod
    def apply_edits(edits: tuple[tuple[str, int, int], ...], source_seq: list[str], target_seq: list[str]) -> list[str]:
        edited_seq: list[str] = list(source_seq)
        indices: list[int] = list(range(len(source_seq)))
        for edit in edits:
            if edit[0] == 'insert':
                if edit[1] == indices[-1] + 1:
                    edited_seq.append(target_seq[edit[2]])
                else:
                    actual_index = indices.index(edit[1])
                    edited_seq.insert(actual_index, target_seq[edit[2]])
                    indices.insert(actual_index, -100)
            elif edit[0] == 'replace':
                actual_index = indices.index(edit[1])
                edited_seq[actual_index] = target_seq[edit[2]]
            elif edit[0] == 'delete':
                actual_index = indices.index(edit[1])
                edited_seq[actual_index] = '<DEL>'
            else:
                raise ValueError('Invalid edit.')
        res_seq = []
        for token in edited_seq:
            if token != '<DEL>':
                res_seq.append(token)
        return res_seq

    @staticmethod
    def evaluate_annos(system_annos: list[Annotation], gold_annos: list[Annotation], method='cherrant') -> dict[
        Annotation, float]:
        """
        Evaluate the scores of the system annotations with respect to the gold ones.
        Allowed methods are 'cherrant' and 'm2scorer'.
        We strongly recommend to avoid using the 'm2scorer' method.
        It is too slow during worker selection, costing 20 hours for a single run.
        This is because m2scorer reads from files, leading to enormous IO cost of time.
        """
        anno2score: dict[Annotation, float] = {}
        for system_anno, gold_anno in zip(system_annos, gold_annos):
            # if system_anno.is_gold:
            #     system_anno.score = 1
            #     continue
            if method == 'cherrant':
                anno2score[system_anno] = cherrant_scores(
                    [system_anno.to_cherrant_format()],
                    [gold_anno.to_cherrant_format()]
                )[2]
            elif method == 'm2scorer':
                system_file_name = f'out/tempo/system-s{system_anno.sent.id}-w{system_anno.worker.id}' \
                                   f'-{datetime.now()}-{random.random()}.txt'
                gold_file_name = f'out/tempo/gold-s{system_anno.sent.id}-w{system_anno.worker.id}' \
                                 f'-{datetime.now()}-{random.random()}.txt'
                with open(system_file_name, 'w') as system_writer, open(gold_file_name, 'w') as gold_writer:
                    system_writer.write(f'{system_anno.to_system_format()}\n')
                    gold_writer.write(f'{gold_anno.to_gold_format()}\n\n')
                res_lines = os.popen(
                    f'python2 m2scorer/m2scorer.py \"{system_file_name}\" \"{gold_file_name}\"').readlines()
                os.remove(system_file_name)
                os.remove(gold_file_name)
                # system_anno.score = float(res_lines[-1].strip())
                anno2score[system_anno] = float(res_lines[-1].strip())
        return anno2score

    def augment_sent(self, sent: Sentence):
        worker2alternatives: dict[Worker, Alternatives] = {}

        # Find all possible edit sequences from existing annotations as Annotations
        possible_annos: list[Annotation] = []
        applied_edit_seqs = []
        for anno in self.original_dataset.sent2annos[sent]:
            if not anno.edit_seq and anno.edit_seq not in applied_edit_seqs:
                applied_edit_seqs.append(anno.edit_seq)
                edited_anno = Annotation(worker=Worker(-1),
                                         content=anno.content,
                                         edit_num=0,
                                         sent=sent)
                anno2score = self.evaluate_annos([edited_anno], [self.original_dataset.sent2gold_anno[sent]])
                edited_anno.score = anno2score[edited_anno]
                possible_annos.append(edited_anno)
                continue
            for length in range(1, len(anno.edit_seq) + 1):
                # We cannot afford the time cost to find all combinations if the edit sequence is longer than 10.
                if len(anno.edit_seq) > 10:
                    edit_combos = [capped_edit_seq + anno.edit_seq[10:]
                                   for capped_edit_seq in itertools.combinations(anno.edit_seq[:10], 10)]
                else:
                    edit_combos = itertools.combinations(anno.edit_seq, length)
                for edit_seq in edit_combos:
                    if edit_seq not in applied_edit_seqs:
                        applied_edit_seqs.append(edit_seq)
                        edited_seq = self.apply_edits(edit_seq, sent.content, anno.content)
                        edited_anno = Annotation(worker=Worker(-1),
                                                 content=edited_seq,
                                                 edit_num=len(edit_seq),
                                                 sent=sent)
                        anno2score = self.evaluate_annos([edited_anno], [self.original_dataset.sent2gold_anno[sent]])
                        edited_anno.score = anno2score[edited_anno]
                        possible_annos.append(edited_anno)
        possible_annos = sorted(possible_annos, key=lambda x: x.score)  # Ascending order

        for worker in self.original_dataset.workers:
            # Real annotation exists.
            if sent in self.original_dataset.worker2sents:
                real_anno = self.original_dataset.get_anno(sent, worker)
                worker2alternatives[worker] = Alternatives(higher_anno=real_anno,
                                                           lower_anno=real_anno,
                                                           closest_anno=real_anno)
                continue
            # Selected from generated annotations.
            highest_lower_anno = possible_annos[0]
            for anno in possible_annos:
                if anno.score <= self.original_dataset.worker2avg_score[worker]:
                    highest_lower_anno = anno
            lowest_higher_anno = possible_annos[-1]
            for anno in reversed(possible_annos):
                if anno.score >= self.original_dataset.worker2avg_score[worker]:
                    lowest_higher_anno = anno
            worker2alternatives[worker] = Alternatives(higher_anno=lowest_higher_anno,
                                                       lower_anno=highest_lower_anno,
                                                       closest_anno=highest_lower_anno)
        return {
            'sent': sent,
            'worker2alternatives': worker2alternatives,
        }

    def augment_dataset(self, processes: int = 8):
        self.augmented_dataset.workers = list(self.original_dataset.workers)
        self.augmented_dataset.sents = list(self.original_dataset.sents)

        if processes == 1:
            results = []
            for sent in tqdm(self.original_dataset.sents):
                results.append(self.augment_sent(sent))
        else:
            with mp.Pool(processes=processes) as pool:
                results = list(tqdm(pool.imap(self.augment_sent, self.original_dataset.sents),
                                    total=len(self.original_dataset.sents)))

        worker2current_scores: dict[Worker, list[float]] = defaultdict(list)

        for result in results:
            # Unpack result
            sent: Sentence = result['sent']
            worker2alternatives: dict[Worker, Alternatives] = result['worker2alternatives']

            for worker, alternatives in worker2alternatives.items():
                # Initialization
                if not worker2current_scores[worker]:
                    self.augmented_dataset.annos.append(Annotation(sent=sent,
                                                                   worker=worker,
                                                                   content=alternatives.higher_anno.content,
                                                                   edit_num=alternatives.higher_anno.edit_num,
                                                                   is_gold=alternatives.higher_anno.is_gold,
                                                                   score=alternatives.higher_anno.score))
                    worker2current_scores[worker].append(alternatives.higher_anno.score)
                    continue
                # After then
                if np.average(worker2current_scores[worker]) < self.original_dataset.worker2avg_score[worker]:
                    self.augmented_dataset.annos.append(Annotation(sent=sent,
                                                                   worker=worker,
                                                                   content=alternatives.higher_anno.content,
                                                                   edit_num=alternatives.higher_anno.edit_num,
                                                                   is_gold=alternatives.higher_anno.is_gold,
                                                                   score=alternatives.higher_anno.score))
                    worker2current_scores[worker].append(alternatives.higher_anno.score)
                else:
                    self.augmented_dataset.annos.append(Annotation(sent=sent,
                                                                   worker=worker,
                                                                   content=alternatives.lower_anno.content,
                                                                   edit_num=alternatives.lower_anno.edit_num,
                                                                   is_gold=alternatives.lower_anno.is_gold,
                                                                   score=alternatives.lower_anno.score))
                    worker2current_scores[worker].append(alternatives.lower_anno.score)

        with open(f'data/cache/aug-data-yaclc-s{len(self.augmented_dataset.sents)}.pkl', 'wb') as writer:
            pickle.dump(self.augmented_dataset, writer)

        with open(f'out/stats/aug-stats-yaclc-s{len(self.augmented_dataset.sents)}.csv', 'w') as writer:
            writer.write('Worker ID, real F_0.5, aug F_0.5\n')
            for worker in self.augmented_dataset.workers:
                writer.write(f'{worker.id}, '
                             f'{self.original_dataset.worker2avg_score[worker]}, '
                             f'{self.augmented_dataset.worker2avg_score[worker]}\n')


def print_dataset_stats(dataset_path: str = 'data/cache/ori-data-yaclc.pkl'):
    """
    Run with:

    python -m src.data_augmentor print_dataset_stats --dataset_path='data/cache/aug-data-yaclc-s32124.pkl'
    """
    dataset = YACLCDataset()
    dataset.read_data(use_cache=True,
                      cache_path=dataset_path)

    print('YACLC')

    worker2anno_num: dict[int, int] = {worker.id: len(annos) for worker, annos in dataset.worker2annos.items()}

    worker2anno_num = dict(sorted(worker2anno_num.items(), key=lambda x: x[1], reverse=True))

    # print(worker2anno_num.keys())
    for worker, num in worker2anno_num.items():
        print(f'{worker}\t{num}')
        # print(f'{num}')

    print(f'worker num: {len(worker2anno_num)}')
    print(f'sent num: {len(dataset.sents)}')

    print(f'max: {max(worker2anno_num.values())}')
    print(f'min: {min(worker2anno_num.values())}')
    print(f'range: {max(worker2anno_num.values()) - min(worker2anno_num.values())}')
    print(f'mean: {mean(worker2anno_num.values())}')
    print(f'median: {median(worker2anno_num.values())}')
    print(f'SD: {stdev(worker2anno_num.values())}')
    print(f'variance: {variance(worker2anno_num.values())}')
    print(f'CV: {stdev(worker2anno_num.values()) / mean(worker2anno_num.values()):.02%}')


def augment_dataset(dataset_path: str = 'data/cache/ori-data-yaclc.pkl', num_processes: int = 8):
    dataset = YACLCDataset()
    dataset.read_data(use_cache=True, cache_path=dataset_path)

    augmentor = YACLCDataAugmentor(original_dataset=dataset)
    augmentor.augment_dataset(processes=num_processes)


if __name__ == "__main__":
    fire.Fire()
