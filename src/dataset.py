import os.path
import pickle
import json
import logging
from collections import defaultdict
from functools import cached_property, lru_cache
from typing import NoReturn

import jieba
import numpy as np
from tqdm import tqdm

from src.data_structure import Annotation, Sentence, Worker
from src.agreement import fleiss_kappa

jieba.setLogLevel(logging.CRITICAL)


class Dataset:

    def __init__(self):
        self.sents: list[Sentence] = []
        self.workers: list[Worker] = []
        self.annos: list[Annotation] = []

    @cached_property
    def worker_ids(self) -> list[int]:
        return [worker.id for worker in self.workers]

    @cached_property
    def sent2annos(self) -> dict[Sentence, list[Annotation]]:
        result = defaultdict(list)
        for anno in self.annos:
            result[anno.sent].append(anno)
        return result

    @cached_property
    def sent2crowd_annos(self) -> dict[Sentence, list[Annotation]]:
        result = defaultdict(list)
        for anno in self.annos:
            if not anno.is_gold:
                result[anno.sent].append(anno)
        return result

    @cached_property
    def sent2gold_anno(self) -> dict[Sentence, Annotation]:
        result = {}
        for anno in self.annos:
            if anno.is_gold:
                result[anno.sent] = anno
        return result

    @cached_property
    def worker2annos(self) -> dict[Worker, list[Annotation]]:
        result = defaultdict(list)
        for anno in self.annos:
            result[anno.worker].append(anno)
        return result

    @cached_property
    def worker2sents(self) -> dict[Worker, list[Sentence]]:
        result = defaultdict(list)
        for anno in self.annos:
            result[anno.worker].append(anno.sent)
        return result

    @cached_property
    def worker2avg_score(self) -> dict[Worker, float]:
        result = {}
        for worker, annos in self.worker2annos.items():
            result[worker] = np.average([anno.score for anno in annos])
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    def best_step_score(self, num_workers: int) -> float:
        result = 0
        for worker, score in list(self.worker2avg_score.items())[:num_workers]:
            result += score
        return result

    @cached_property
    def id2sent(self) -> dict[int, Sentence]:
        return {sent.id: sent for sent in self.sents}

    def get_anno(self, sent: Sentence, worker: Worker) -> Annotation:
        for anno in self.sent2annos[sent]:
            if anno.worker == worker:
                return anno
        raise ValueError('Annotation not found.')

    def read_data(self, data_path: str) -> NoReturn:
        pass

    def dump_data(self, data_path: str = f'data/cache/ori-data-yaclc.pkl') -> NoReturn:
        with open(data_path, 'wb') as writer:
            pickle.dump(self, writer)


class YACLCDataset(Dataset):

    def read_data(self, data_path: str = '', sent_num_limit: int = -1,
                  use_cache: bool = True, cache_path: str = 'data/cache/ori-data-yaclc.pkl') -> NoReturn:
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as reader:
                cache: YACLCDataset = pickle.load(reader)
            self.workers = cache.workers
            self.sents = cache.sents
            self.annos = cache.annos
            return
        with open(data_path, 'r') as data_file:
            data = json.load(data_file)
        for article_data in data:
            for sent_data in article_data['sentences']:
                sent = Sentence(sent_data['sentence_id'], jieba.lcut(sent_data['sentence_text']))
                self.sents.append(sent)

                gold_anno = None
                max_worker_num = 0

                for anno_data in sent_data['sentence_grammatical_annos']:
                    for worker_id in anno_data['usr_ids']:
                        worker = Worker(worker_id)
                        if worker not in self.workers:
                            self.workers.append(worker)

                        anno_content = jieba.lcut(anno_data['correction'])
                        anno = Annotation(worker=worker,
                                          sent=sent,
                                          content=anno_content,
                                          edit_num=anno_data['edits_count'])
                        self.annos.append(anno)

                        if anno_data['usr_count'] > max_worker_num:
                            max_worker_num = anno_data['usr_count']
                            gold_anno = anno

                gold_anno.is_gold = True

                if len(self.sents) >= sent_num_limit != -1:
                    return

    def do_statistics(self, data_path: str, sent_num_limit: int = -1,
                      use_cache: bool = True, cache_path: str = 'data/cache/ori-data-yaclc.pkl') -> list[tuple[int, int]]:
        with open(data_path, 'r') as data_file:
            data = json.load(data_file)

        statistics: list[tuple[int, int]] = []  # [(max_worker_num, total_worker_num), ...]

        for article_data in data:
            for sent_data in article_data['sentences']:
                sent = Sentence(sent_data['sentence_id'], jieba.lcut(sent_data['sentence_text']))
                self.sents.append(sent)

                gold_anno = None
                max_worker_num = 0
                total_worker_num = 0

                for anno_data in sent_data['sentence_grammatical_annos']:
                    total_worker_num += anno_data['usr_count']

                    # for worker_id in anno_data['usr_ids']:
                    #     worker = Worker(worker_id)
                    #     if worker not in self.workers:
                    #         self.workers.append(worker)
                    #
                    #     anno_content = jieba.lcut(anno_data['correction'])
                    #     anno = Annotation(worker=worker,
                    #                       sent=sent,
                    #                       content=anno_content,
                    #                       edit_num=anno_data['edits_count'])
                    #     self.annos.append(anno)

                    if anno_data['usr_count'] > max_worker_num:
                        max_worker_num = anno_data['usr_count']

                # gold_anno.is_gold = True

                statistics.append((max_worker_num, total_worker_num))
                # print(f'sent: {sent.id}, max: {max_worker_num}, total: {total_worker_num}')

                if len(self.sents) >= sent_num_limit != -1:
                    return statistics

        return statistics

    def get_agreement(self) -> dict[Sentence, float]:
        sent2agreement: dict[Sentence, float] = {}

        for sent, annos in tqdm(self.sent2annos.items()):
            annos_formatted: list[list[str]] = [[str(edit) for edit in anno.edit_seq] for anno in annos]
            max_edit_num = max([len(l) for l in annos_formatted])
            # print([len(l) for l in annos_formatted])
            for l in annos_formatted:
                while len(l) < max_edit_num:
                    l.append('O')
            # print(annos_formatted)
            sent2agreement[sent] = fleiss_kappa(annos_formatted)

        return sent2agreement


class OEIDataset(Dataset):
    pass


