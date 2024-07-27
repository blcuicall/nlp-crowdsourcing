import copy
import json
import os.path
import random
from collections import defaultdict, namedtuple
from functools import reduce
from typing import List, Union, NoReturn, Tuple, Dict, Optional
import multiprocessing as mp
import warnings
import logging

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from metrics import Metrics

# Presets
warnings.filterwarnings('error')
# np.seterr(invalid='ignore', divide='ignore')
jieba.setLogLevel(logging.CRITICAL)
random.seed(125869)

Scores = namedtuple('Scores', ('precisions', 'recalls', 'f1s'))

Alternatives = namedtuple('Alternatives', ('higher_tags', 'higher_f1',
                                           'lower_tags', 'lower_f1',
                                           'closest'))


class Utils:

    def __init__(self, annotation_num_per_sentence=1,
                 allow_fake_annotations=False,
                 only_fake_annotations=False,
                 metrics_type='span_exact',
                 acceptable_fake_anno_error=0.025,
                 max_possible_fake_anno_num=30000,
                 use_fake_annotation_cache=False,
                 memory=None):
        self.annotation_num_per_sentence = annotation_num_per_sentence
        if metrics_type not in ['span_exact', 'span_proportional', 'token', 'pearson']:
            raise ValueError(
                f'Wrong metrics level given: {metrics_type}. Should be \'span_exact\', \'span_exact\', \'token\' or \'pearson\'.')

        self.metrics_type = metrics_type
        self.metrics = Metrics(metrics_type=metrics_type)
        self.allow_fake_annotations = allow_fake_annotations
        self.only_fake_annotations = only_fake_annotations

        self.acceptable_fake_anno_error = acceptable_fake_anno_error
        self.max_possible_fake_anno_num = max_possible_fake_anno_num
        self.error2times: dict[float, int] = {}

        self.cache_filename = 'out/annotation_matrix.conll'
        if self.metrics_type == 'span_exact':
            self.cache_filename += '.exact'
        elif self.metrics_type == 'span_proportional':
            self.cache_filename += '.prop'
        elif self.metrics_type == 'token':
            self.cache_filename += '.token'
        self.cache_filename += '.json'

        with open('out/debug_logs.txt', 'w') as _:
            pass
        with open('out/fake_annotations.txt', 'w') as _:
            pass

        self.tag2index = {
            'O': 0,
            'B-PER': 1,
            'I-PER': 2,
            'B-ORG': 3,
            'I-ORG': 4,
            'B-LOC': 5,
            'I-LOC': 6,
            'B-MISC': 7,
            'I-MISC': 8
        }

        self.tag_types = list(set(key.removeprefix('B-').removeprefix('I-') for key in self.tag2index.keys()))

        # Data example:
        # data = [{
        #     'id': 123,
        #     'filename': 'ABC.txt',
        #     'idx_in_file': 2,
        #     'words': ['Hello', 'world', '!'],
        #     'expert_annotation': ['B-MISC', 'I-MISC', 'O'],
        #     'worker_ids': [12, 43],
        #     'worker_annotations': [
        #         ['O', 'B-MISC', 'I-MISC'],
        #         ['O', 'B-MISC', 'O']
        #     ]
        # }, ...]
        with open('data/conll-2003/collected.json', 'r') as data_file:
            self.data = json.load(data_file)

        # Load memory to keep experiments consistent.
        if memory:
            self.data = memory['data']

        # Sentence split into words
        self.sentences: dict[int, list[str]] = {}
        for sentence in self.data:
            self.sentences[sentence['id']] = sentence['words']

        # Expert annotation tags of each sentence
        # Example:
        # sent2silver_tags = {
        #     10964: ['O', 'O', 'B-LOC', 'I-LOC'],
        #     10943: ['B-LOC', 'I-LOC', 'O', 'O'],
        # }
        self.sent2silver_tags: dict[int, list[str]] = {
            sentence['id']: sentence['expert_annotation']
            for sentence in self.data
        }
        # Load memory to keep experiments consistent.
        if memory:
            self.sent2silver_tags = memory['sent2silver_tags']

        # All annotation tags of each sentence
        # Example:
        # sent2all_tags = {
        #     10964: {1: ['O', 'O', 'B-LOC', 'I-LOC'], 5: ['O', 'O', 'B-LOC', 'I-LOC']},
        #     4530: {45: ['B-LOC', 'I-LOC', 'O', 'O'], 33: ['O', 'O', 'B-LOC', 'I-LOC']},
        # }
        self.sent2all_tags: dict[int, dict[int, list[str]]] = {
            sentence['id']: {
                worker_id: worker_annotation
                for worker_id, worker_annotation in zip(sentence['worker_ids'], sentence['worker_annotations'])
            }
            for sentence in self.data
        }

        # Find out which workers annotated each sentence
        self.sent2workers: dict[int, list[int]] = {
            sent_id: list(worker_id2tags.keys())
            for sent_id, worker_id2tags in self.sent2all_tags.items()
        }

        # Find out which sentences are annotated by each worker
        self.worker2sents: dict[int, list[int]] = {}
        for sent, worker_ids in self.sent2workers.items():
            for worker_id in worker_ids:
                if worker_id not in self.worker2sents.keys():
                    self.worker2sents[worker_id] = []
                self.worker2sents[worker_id].append(sent)

        # All worker ids
        self.worker_ids = list(self.worker2sents.keys())

        # Mean precision of workers
        self.worker2mean_precision = {}
        self.worker2mean_prop_precision = {}
        self.worker2mean_exact_precision = {}
        for worker, sents in self.worker2sents.items():
            worker_annotations = []
            silver_annotations = []
            for sentence in sents:
                worker_annotations.append(self.sent2all_tags[sentence][worker])
                silver_annotations.append(self.sent2silver_tags[sentence])
            self.worker2mean_precision[worker] = self.metrics.precision_score(silver_annotations, worker_annotations)
            self.worker2mean_prop_precision[worker] = self.metrics.precision_score(silver_annotations, worker_annotations, metrics_type='span_proportional')
            self.worker2mean_exact_precision[worker] = self.metrics.precision_score(silver_annotations, worker_annotations, metrics_type='span_exact')

        # Mean recall of workers
        self.worker2mean_recall = {}
        self.worker2mean_prop_recall = {}
        self.worker2mean_exact_recall = {}
        for worker, sents in self.worker2sents.items():
            worker_annotations = []
            silver_annotations = []
            for sentence in sents:
                worker_annotations.append(self.sent2all_tags[sentence][worker])
                silver_annotations.append(self.sent2silver_tags[sentence])
            self.worker2mean_recall[worker] = self.metrics.recall_score(silver_annotations, worker_annotations)
            self.worker2mean_prop_recall[worker] = self.metrics.recall_score(silver_annotations, worker_annotations, metrics_type='span_proportional')
            self.worker2mean_exact_recall[worker] = self.metrics.recall_score(silver_annotations, worker_annotations, metrics_type='span_exact')

        # Mean F-1 of workers
        self.worker2mean_f1: dict[int, float] = {}
        self.worker2mean_prop_f1: dict[int, float] = {}
        self.worker2mean_exact_f1: dict[int, float] = {}
        for worker, sents in self.worker2sents.items():
            worker_annotations = []
            silver_annotations = []
            for sentence in sents:
                worker_annotations.append(self.sent2all_tags[sentence][worker])
                silver_annotations.append(self.sent2silver_tags[sentence])
            self.worker2mean_f1[worker] = self.metrics.f1_score(silver_annotations, worker_annotations)
            self.worker2mean_prop_f1[worker] = self.metrics.f1_score(silver_annotations, worker_annotations, metrics_type='span_proportional')
            self.worker2mean_exact_f1[worker] = self.metrics.f1_score(silver_annotations, worker_annotations, metrics_type='span_exact')

        # Mean cohens kappa of workers
        self.worker2mean_cohens_kappa = {}
        for worker, sents in self.worker2sents.items():
            kappas = []
            for sentence in sents:
                kappas.append(self.metrics.cohens_kappa([self.sent2all_tags[sentence][worker],
                                                         self.sent2silver_tags[sentence]]))
            self.worker2mean_cohens_kappa[worker] = np.mean(kappas)

        if self.metrics_type == 'pearson':
            # Mean pcc of workers
            self.worker2mean_pcc = {}
            for worker, sents in self.worker2sents.items():
                worker_annotations = []
                silver_annotations = []
                for sentence in sents:
                    worker_annotations.append(self.sent2all_tags[sentence][worker])
                    silver_annotations.append(self.sent2silver_tags[sentence])
                self.worker2mean_pcc[worker] = self.metrics.pearson_corr_coef(silver_annotations, worker_annotations)

        # Key: worker id
        # Value: ids of sentences annotated by this user in the train data but not yet annotated in the CMAB process
        self.worker2unannotated_sents: dict[int, list[int]] = {}
        for sent, worker_ids in self.sent2workers.items():
            for worker_id in worker_ids:
                if worker_id not in self.worker2unannotated_sents.keys():
                    self.worker2unannotated_sents[worker_id] = []
                self.worker2unannotated_sents[worker_id].append(sent)

        # To add randomness in task assignment, shuffle sentence ids.
        sent_ids_assign = list(self.sentences.keys())
        random.shuffle(sent_ids_assign)

        if allow_fake_annotations:
            for worker_id in self.worker_ids:
                for sent_id in sent_ids_assign:
                    if sent_id not in self.worker2unannotated_sents[worker_id]:
                        self.worker2unannotated_sents[worker_id].append(sent_id)

        # If one sentence could have multiple annotations, duplicate workers' unannotated_sents
        for worker, sents in self.worker2unannotated_sents.items():
            multiplied_sents = []
            for sent in sents:
                multiplied_sents.extend([sent] * annotation_num_per_sentence)
            self.worker2unannotated_sents[worker] = multiplied_sents

        self.fake_precision_errors: list[float] = []
        self.fake_recall_errors: list[float] = []
        self.fake_f1_errors: list[float] = []
        self.fake_pcc_errors: list[float] = []

        # Check if there is available cache
        self.use_fake_annotation_cache = use_fake_annotation_cache
        self.annotation_matrix: dict[int, dict[int, list[str]]] = {}
        if os.path.exists(self.cache_filename) and self.use_fake_annotation_cache:
            with open(self.cache_filename, 'r') as anno_mat:
                data = json.load(anno_mat)
            self.annotation_matrix = {
                int(sent_id): {
                    int(worker_id): tags
                    for worker_id, tags in worker2tags.items()
                }
                for sent_id, worker2tags in data.items()
            }

    def get_fake_annotation(self, sentence_id: int, worker_id: int,
                            preset_fake_annotations: Optional[list[list[str]]] = None,
                            preset_scores: Optional[Scores] = None,
                            preset_exact_scores: Optional[Scores] = None) \
            -> tuple[list[str], list[list[str]], Scores, Scores, Alternatives]:
        # Find all possible annotations
        characters = self.sentences[sentence_id]
        words = self.sentences[sentence_id]
        silver_tags = self.sent2silver_tags[sentence_id]
        if preset_fake_annotations:
            possible_fake_annotations: list[list[str]] = preset_fake_annotations
        elif self.use_fake_annotation_cache:
            possible_fake_annotations: list[list[str]] = [self.annotation_matrix[sentence_id][worker_id]]
        else:
            possible_fake_annotations: list[list[str]] = self.get_possible_fake_annotations(characters, words,
                                                                                            silver_tags)
        # possible_fake_annotations: list[list[str]] = self.get_possible_fake_annotations_random_gen(characters, words)

        # Calculate scores
        if preset_scores:
            precisions = preset_scores.precisions
            recalls = preset_scores.recalls
            f1s = preset_scores.f1s
            pccs = []
            scores = preset_scores
        else:
            precisions = []
            recalls = []
            f1s = []
            pccs = []
            silver_annotation = self.sent2silver_tags[sentence_id]
            for i, fake_annotation in enumerate(possible_fake_annotations):
                precisions.append(self.metrics.precision_score([silver_annotation], [fake_annotation]))
                recalls.append(self.metrics.recall_score([silver_annotation], [fake_annotation]))
                f1s.append(self.metrics.f1_score([silver_annotation], [fake_annotation]))
                if self.metrics_type == 'pearson':
                    pccs.append(self.metrics.pearson_corr_coef([silver_annotation], [fake_annotation]))
            scores = Scores(precisions, recalls, f1s)

        # Calculate span-level exact scores
        if preset_exact_scores:
            exact_precisions = preset_exact_scores.precisions
            exact_recalls = preset_exact_scores.recalls
            exact_f1s = preset_exact_scores.f1s
            exact_scores = preset_exact_scores
        else:
            exact_precisions = []
            exact_recalls = []
            exact_f1s = []
            silver_annotation = self.sent2silver_tags[sentence_id]
            for i, fake_annotation in enumerate(possible_fake_annotations):
                exact_precisions.append(self.metrics.precision_score([silver_annotation], [fake_annotation], metrics_type='span_exact'))
                exact_recalls.append(self.metrics.recall_score([silver_annotation], [fake_annotation], metrics_type='span_exact'))
                exact_f1s.append(self.metrics.f1_score([silver_annotation], [fake_annotation], metrics_type='span_exact'))
            exact_scores = Scores(exact_precisions, exact_recalls, exact_f1s)

        f1_errors = [f1 - self.worker2mean_f1[worker_id] for f1 in f1s]
        pos_f1_errors = list(filter(lambda x: x >= 0, f1_errors))
        neg_f1_errors = list(filter(lambda x: x <= 0, f1_errors))
        closest_higher_index = f1_errors.index(min(pos_f1_errors)) if pos_f1_errors \
            else f1_errors.index(max(neg_f1_errors))
        closest_lower_index = f1_errors.index(max(neg_f1_errors)) if neg_f1_errors \
            else f1_errors.index(min(pos_f1_errors))

        # Select the one with highest exact f1 score
        for i, f1 in enumerate(f1s):
            if f1 == f1s[closest_higher_index] and exact_f1s[i] > exact_f1s[closest_higher_index]:
                closest_higher_index = i
            if f1 == f1s[closest_lower_index] and exact_f1s[i] > exact_f1s[closest_lower_index]:
                closest_lower_index = i

        if abs(f1_errors[closest_higher_index]) < abs(f1_errors[closest_lower_index]):
            closest = 'higher'
        else:
            closest = 'lower'

        alternatives = Alternatives(higher_tags=possible_fake_annotations[closest_higher_index],
                                    higher_f1=f1s[closest_higher_index],
                                    lower_tags=possible_fake_annotations[closest_lower_index],
                                    lower_f1=f1s[closest_lower_index],
                                    closest=closest)

        # acceptable_indices = []
        # current_acceptable_error = self.acceptable_fake_anno_error
        # while not acceptable_indices:
        #     for i, _ in enumerate(possible_fake_annotations):
        #         if abs(precisions[i] - self.worker2mean_precision[worker_id]) < current_acceptable_error and \
        #                 abs(recalls[i] - self.worker2mean_recall[worker_id]) < current_acceptable_error:
        #             acceptable_indices.append(i)
        #     if not acceptable_indices:
        #         current_acceptable_error *= 2

        # if current_acceptable_error not in self.error2times.keys():
        #     self.error2times[current_acceptable_error] = 0
        # self.error2times[current_acceptable_error] += 1

        # final_index = random.choice(acceptable_indices)

        final_index = random.choice([closest_higher_index, closest_lower_index])
        final_fake_annotation = possible_fake_annotations[final_index]
        self.fake_precision_errors.append(abs(precisions[final_index] - self.worker2mean_precision[worker_id]))
        self.fake_recall_errors.append(abs(recalls[final_index] - self.worker2mean_recall[worker_id]))
        self.fake_f1_errors.append(abs(f1s[final_index] - self.worker2mean_f1[worker_id]))
        if self.metrics_type == 'pearson':
            self.fake_pcc_errors.append(abs(pccs[final_index] - self.worker2mean_pcc[worker_id]))

        with open('out/fake_annotations.txt', 'a') as fake_annotations:
            fake_annotations.write(
                f'{sentence_id}: {worker_id}: {f1s[final_index]}: {self.fake_f1_errors[-1]}: {final_fake_annotation}\n')

        return final_fake_annotation, possible_fake_annotations, scores, exact_scores, alternatives

    def process(self, sent_id: int):
        worker2tags: dict[int, list[str]] = {}
        num_possibilities: int = 0
        worker2alternatives = {}
        possible_fake_annotations = None
        scores = None
        exact_scores = None
        for worker_id in self.worker_ids:
            # Real annotation exists.
            if worker_id in self.sent2workers[sent_id]:
                worker2tags[worker_id] = self.sent2all_tags[sent_id][worker_id]
                f1 = self.metrics.f1_score([self.sent2silver_tags[sent_id]], [worker2tags[worker_id]])
                worker2alternatives[worker_id] = Alternatives(higher_tags=worker2tags[worker_id],
                                                              higher_f1=f1,
                                                              lower_tags=worker2tags[worker_id],
                                                              lower_f1=f1,
                                                              closest='higher')
                continue
            # Precalculate all possible fake annotations and scores
            if not possible_fake_annotations:
                fake_annotation, possible_fake_annotations, \
                scores, exact_scores, alternatives = self.get_fake_annotation(sent_id, worker_id)
            else:
                fake_annotation, _, _, _, alternatives = self.get_fake_annotation(sent_id, worker_id,
                                                                                  possible_fake_annotations,
                                                                                  scores, exact_scores)
            worker2tags[worker_id] = fake_annotation
            num_possibilities = len(possible_fake_annotations)
            worker2alternatives[worker_id] = alternatives

        return {
            'sent_id': sent_id,
            'worker2tags': worker2tags,
            'worker2alternatives': worker2alternatives,
            'num_possibilities': num_possibilities,
            'scores': scores
        }

    def cache_annotation_matrix(self):
        matrix: dict[int, dict[int, list[str]]] = {}
        sent2num_possibilities: dict[int, int] = {}
        scores_matrix: dict[int, dict[str, list[float]]] = {}

        with mp.Pool(processes=32) as pool:
            results = list(tqdm(pool.imap(self.process, self.sent2workers.keys()), total=len(self.sent2workers)))

        worker2current_f1s: dict[int, list[float]] = defaultdict(list)

        for result in results:
            # Unpack the result
            sent_id: int = result['sent_id']
            worker2tags: dict[int, list[str]] = result['worker2tags']
            worker2alternatives: dict[int, Alternatives] = result['worker2alternatives']
            scores: Scores = result['scores']

            # Initial values
            matrix[sent_id] = worker2tags

            # Dynamically adjust worker's current f1 to fit its empirical
            for worker_id, tags in worker2tags.items():
                # if worker_id not in worker2current_f1s.keys():
                #     # Might lead to a little bias
                #     worker2current_f1s[worker_id].append(worker2alternatives[worker_id].higher_f1)
                #     matrix[sent_id][worker_id] = worker2alternatives[worker_id].higher_tags
                #     continue
                # if np.mean(worker2current_f1s[worker_id]) > self.worker2mean_f1[worker_id]:
                #     worker2current_f1s[worker_id].append(worker2alternatives[worker_id].lower_f1)
                #     matrix[sent_id][worker_id] = worker2alternatives[worker_id].lower_tags
                # else:
                #     worker2current_f1s[worker_id].append(worker2alternatives[worker_id].higher_f1)
                #     matrix[sent_id][worker_id] = worker2alternatives[worker_id].higher_tags

                if worker2alternatives[worker_id].closest == 'lower':
                    worker2current_f1s[worker_id].append(worker2alternatives[worker_id].lower_f1)
                    matrix[sent_id][worker_id] = worker2alternatives[worker_id].lower_tags
                else:
                    worker2current_f1s[worker_id].append(worker2alternatives[worker_id].higher_f1)
                    matrix[sent_id][worker_id] = worker2alternatives[worker_id].higher_tags

            sent2num_possibilities[sent_id] = result['num_possibilities']
            scores_matrix[sent_id] = {
                'precisions': scores.precisions,
                'recalls': scores.recalls,
                'f1s': scores.f1s,
            }

        with open(self.cache_filename, 'w') as out:
            json.dump(matrix, out, sort_keys=True)

        with open(self.cache_filename.replace('.json', '.num_p.json'), 'w') as out:
            json.dump(sent2num_possibilities, out, sort_keys=True)

        with open(self.cache_filename.replace('.json', '.scores.json'), 'w') as out:
            json.dump(scores_matrix, out, sort_keys=True)

        worker_id2fake_prop_precisions = defaultdict(list)
        worker_id2fake_prop_recalls = defaultdict(list)
        worker_id2fake_prop_f1s = defaultdict(list)
        worker_id2fake_exact_precisions = defaultdict(list)
        worker_id2fake_exact_recalls = defaultdict(list)
        worker_id2fake_exact_f1s = defaultdict(list)
        for sent_id, worker_id2tags in matrix.items():
            for worker_id, tags in worker_id2tags.items():
                scores: dict[str, float] = self.metrics.span_proportional_scores([self.sent2silver_tags[sent_id]], [tags])
                worker_id2fake_prop_precisions[worker_id].append(scores['precision'])
                worker_id2fake_prop_recalls[worker_id].append(scores['recall'])
                worker_id2fake_prop_f1s[worker_id].append(scores['F1'])
                worker_id2fake_exact_precisions[worker_id].append(self.metrics.precision_score([self.sent2silver_tags[sent_id]], [tags], metrics_type='span_exact'))
                worker_id2fake_exact_recalls[worker_id].append(self.metrics.recall_score([self.sent2silver_tags[sent_id]], [tags], metrics_type='span_exact'))
                worker_id2fake_exact_f1s[worker_id].append(self.metrics.f1_score([self.sent2silver_tags[sent_id]], [tags], metrics_type='span_exact'))

        worker_id2real_f1 = dict(sorted(self.worker2mean_f1.items(), key=lambda item: item[1], reverse=True))

        with open(self.cache_filename.replace('.json', '.report.csv'), 'w') as out:
            out.write(f', Prop, Prop, Prop, Prop, Prop, Prop, Exact, Exact, Exact, Exact, Exact, Exact\n')
            out.write(f'Worker ID, Real P, Fake P, Real R, Fake R, Real F1, Fake F1, Real P, Fake P, Real R, Fake R, Real F1, Fake F1\n')
            for worker_id in worker_id2real_f1.keys():
                out.write(f'{worker_id}, '
                          f'{self.worker2mean_prop_precision[worker_id]:.02%}, '
                          f'{np.mean(worker_id2fake_prop_precisions[worker_id]):.02%}, '
                          f'{self.worker2mean_prop_recall[worker_id]:.02%}, '
                          f'{np.mean(worker_id2fake_prop_recalls[worker_id]):.02%}, '
                          f'{self.worker2mean_prop_f1[worker_id]:.02%}, '
                          f'{np.mean(worker_id2fake_prop_f1s[worker_id]):.02%}, '
                          f'{self.worker2mean_exact_precision[worker_id]:.02%}, '
                          f'{np.mean(worker_id2fake_exact_precisions[worker_id]):.02%}, '
                          f'{self.worker2mean_exact_recall[worker_id]:.02%}, '
                          f'{np.mean(worker_id2fake_exact_recalls[worker_id]):.02%}, '
                          f'{self.worker2mean_exact_f1[worker_id]:.02%}, '
                          f'{np.mean(worker_id2fake_exact_f1s[worker_id]):.02%}\n')

        print('Error: Times')
        for error, times in sorted(self.error2times.items(), key=lambda item: item[0]):
            print(f'{error}: {times}, {times / sum(self.error2times.values()):.02%}')

    def get_possible_fake_annotations_random_gen(self, words: list[str]):
        possible_annotations: list[list[str]] = [[tag_type] for tag_type in self.tag_types]

        for i, word in enumerate(words):
            if i == 0:
                continue
            new_possible_annotations = []
            for annotation in possible_annotations:
                for tag_type in self.tag_types:
                    new_possible_annotations.append(annotation.copy() + [tag_type])

        return possible_annotations

    def get_possible_fake_annotations(self, characters: list[str], words: list[str], silver_tags: list[str]):
        # Preprocessing
        # Mark characters with corresponding words.
        word_marks: list[int] = []
        for i, word in enumerate(words):
            for _ in list(word):
                word_marks.append(i)
        word_marks.append(-1)

        # Identify the spans.
        class Span:
            def __init__(self, start: Optional[int] = None, end: Optional[int] = None, pos: Optional[str] = None):
                self.start = start  # Span includes the start position
                self.end = end  # Span does not include the end position
                self.pos = pos

            @property
            def length(self):
                return self.end - self.start

            def pos_randomised(self):
                randomised_span = copy.deepcopy(self)
                tag_types = ['PER', 'LOC', 'ORG', 'MISC']
                tag_types.remove(randomised_span.pos)
                randomised_span.pos = random.choice(tag_types)
                return randomised_span

            def __eq__(self, other):
                return self.start == other.start and self.end == other.end and self.pos == other.pos

            def __hash__(self):
                return hash((self.start, self.end, self.pos))

            def __repr__(self):
                return f'{self.pos}-{self.start}-{self.end}'

        silver_spans: list[Span] = []
        is_in_span = False
        current_span = Span()
        for i, tag in enumerate(silver_tags):
            if is_in_span:
                if tag.startswith('I'):
                    continue
                current_span.end = i
                silver_spans.append(current_span)
                if tag == 'O':
                    is_in_span = False
                    current_span = Span()
                elif tag.startswith('B'):
                    current_span = Span(start=i, pos=tag.removeprefix('B-'))
            elif tag.startswith('B'):
                is_in_span = True
                current_span.start = i
                current_span.pos = tag.removeprefix('B-')
        if is_in_span:
            current_span.end = len(silver_tags)
            silver_spans.append(current_span)

        with open('out/debug_logs.txt', 'a') as debug_logs:
            debug_logs.write(f'{"".join(characters)} : {len(silver_spans)}')

        # Modify the spans.
        possible_annotations: list[list[Span]] = []
        for i, silver_span in enumerate(silver_spans):
            if i == 0:  # First span
                # Add the original span
                possible_annotations.append([copy.deepcopy(silver_span)])
                possible_annotations.append([silver_span.pos_randomised()])

                # Decide the borders of shifting and extending
                left_border = 0
                right_border = len(characters) if i == len(silver_spans) - 1 else silver_spans[i + 1].start

                # Shifting left
                shifted_span = copy.deepcopy(silver_span)
                while shifted_span.end > silver_span.start:
                    # Shifting start of the span
                    # Find the new word mark
                    initial_word_mark = word_marks[shifted_span.start]
                    while word_marks[shifted_span.start] == initial_word_mark and shifted_span.start > left_border:
                        shifted_span.start -= 1
                    # Shift to the new word mark
                    new_word_mark = word_marks[shifted_span.start]
                    while word_marks[shifted_span.start] == new_word_mark and shifted_span.start > left_border:
                        shifted_span.start -= 1
                    if word_marks[shifted_span.start] != new_word_mark:
                        shifted_span.start += 1
                    # Shifting end of the span
                    # Find the new word mark
                    initial_word_mark = word_marks[shifted_span.end]
                    while word_marks[shifted_span.end] == initial_word_mark and shifted_span.end > left_border:
                        shifted_span.end -= 1
                    # Shift to the new word mark
                    new_word_mark = word_marks[shifted_span.end]
                    while word_marks[shifted_span.end] == new_word_mark and shifted_span.end > left_border:
                        shifted_span.end -= 1
                    if word_marks[shifted_span.end] != new_word_mark:
                        shifted_span.end += 1
                    possible_annotations.append([shifted_span])
                    shifted_span = copy.deepcopy(shifted_span)

                # Shifting right
                shifted_span = copy.deepcopy(silver_span)
                while shifted_span.start < silver_span.end:
                    # Shifting end of the span
                    # Find and shift to the new word mark
                    initial_word_mark = word_marks[shifted_span.end]
                    while word_marks[shifted_span.end] == initial_word_mark and shifted_span.end < right_border:
                        shifted_span.end += 1
                    # Shifting start of the span
                    # Find and shift to the new word mark
                    initial_word_mark = word_marks[shifted_span.start]
                    while word_marks[shifted_span.start] == initial_word_mark and shifted_span.start < right_border:
                        shifted_span.start += 1
                    possible_annotations.append([shifted_span])
                    shifted_span = copy.deepcopy(shifted_span)

                # Extending left
                extended_span = copy.deepcopy(silver_span)
                while extended_span.start > left_border:
                    # Extending start of the span
                    initial_word_mark = word_marks[extended_span.start]
                    while word_marks[extended_span.start] == initial_word_mark and extended_span.start > left_border:
                        extended_span.start -= 1
                    new_word_mark = word_marks[extended_span.start]
                    while word_marks[extended_span.start] == new_word_mark and extended_span.start > left_border:
                        extended_span.start -= 1
                    if word_marks[extended_span.start] != new_word_mark:
                        extended_span.start += 1
                    possible_annotations.append([extended_span])
                    # possible_annotations.append([extended_span.pos_reversed()])
                    extended_span = copy.deepcopy(extended_span)

                # Extending right
                extended_span = copy.deepcopy(silver_span)
                while extended_span.end < right_border:
                    # Extending end of the span
                    initial_word_mark = word_marks[extended_span.end]
                    while word_marks[extended_span.end] == initial_word_mark and extended_span.end < right_border:
                        extended_span.end += 1
                    possible_annotations.append([extended_span])
                    # possible_annotations.append([extended_span.pos_reversed()])
                    extended_span = copy.deepcopy(extended_span)

                # Shrinking left
                shrunk_span = copy.deepcopy(silver_span)
                while shrunk_span.start < shrunk_span.end:
                    # Shrinking start of the span
                    initial_word_mark = word_marks[shrunk_span.start]
                    while word_marks[
                        shrunk_span.start] == initial_word_mark and shrunk_span.start < shrunk_span.end:
                        shrunk_span.start += 1
                    if shrunk_span.length > 0:
                        possible_annotations.append([shrunk_span])
                        shrunk_span = copy.deepcopy(shrunk_span)

                # Shrinking right
                shrunk_span = copy.deepcopy(silver_span)
                while shrunk_span.end > shrunk_span.start:
                    # Shrinking end of the span
                    initial_word_mark = word_marks[shrunk_span.end]
                    while word_marks[shrunk_span.end] == initial_word_mark and shrunk_span.start < shrunk_span.end:
                        shrunk_span.end -= 1
                    new_word_mark = word_marks[shrunk_span.end]
                    while word_marks[shrunk_span.end] == new_word_mark and shrunk_span.start < shrunk_span.end:
                        shrunk_span.end -= 1
                    if shrunk_span.start < shrunk_span.end:
                        shrunk_span.end += 1
                    if shrunk_span.length > 0:
                        possible_annotations.append([shrunk_span])
                        shrunk_span = copy.deepcopy(shrunk_span)

            else:  # Other spans
                new_possible_annotations: list[list[Span]] = []
                # Go through all existing span sequences
                for annotation in possible_annotations:
                    # Add the original span
                    new_possible_annotations.append(annotation + [copy.deepcopy(silver_span)])
                    new_possible_annotations.append(annotation + [silver_span.pos_randomised()])

                    # Decide the borders of shifting and extending
                    left_border = annotation[-1].end
                    right_border = len(characters) if i == len(silver_spans) - 1 else silver_spans[i + 1].start - 1

                    # Shifting left
                    shifted_span = copy.deepcopy(silver_span)
                    while shifted_span.end > silver_span.start:
                        # Shifting start of the span
                        # Find the new word mark
                        initial_word_mark = word_marks[shifted_span.start]
                        while word_marks[
                            shifted_span.start] == initial_word_mark and shifted_span.start > left_border:
                            shifted_span.start -= 1
                        # Shift to the new word mark
                        new_word_mark = word_marks[shifted_span.start]
                        while word_marks[shifted_span.start] == new_word_mark and shifted_span.start > left_border:
                            shifted_span.start -= 1
                        if word_marks[shifted_span.start] != new_word_mark:
                            shifted_span.start += 1
                        # Shifting end of the span
                        # Find the new word mark
                        initial_word_mark = word_marks[shifted_span.end]
                        while word_marks[shifted_span.end] == initial_word_mark and shifted_span.end > left_border:
                            shifted_span.end -= 1
                        # Shift to the new word mark
                        new_word_mark = word_marks[shifted_span.end]
                        while word_marks[shifted_span.end] == new_word_mark and shifted_span.end > left_border:
                            shifted_span.end -= 1
                        if word_marks[shifted_span.end] != new_word_mark:
                            shifted_span.end += 1
                        new_possible_annotations.append(annotation + [shifted_span])
                        shifted_span = copy.deepcopy(shifted_span)
                        if len(new_possible_annotations) % 100 == 0:
                            with open('out/debug_logs.txt', 'a') as debug_logs:
                                debug_logs.write(f': {len(new_possible_annotations)} ')
                        if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                            break
                    if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                        break

                    # Shifting right
                    shifted_span = copy.deepcopy(silver_span)
                    while shifted_span.start < silver_span.end:
                        # Shifting end of the span
                        # Find and shift to the new word mark
                        initial_word_mark = word_marks[shifted_span.end]
                        while word_marks[shifted_span.end] == initial_word_mark and shifted_span.end < right_border:
                            shifted_span.end += 1
                        # Shifting start of the span
                        # Find and shift to the new word mark
                        initial_word_mark = word_marks[shifted_span.start]
                        while word_marks[
                            shifted_span.start] == initial_word_mark and shifted_span.start < right_border:
                            shifted_span.start += 1
                        new_possible_annotations.append(annotation + [shifted_span])
                        shifted_span = copy.deepcopy(shifted_span)
                        if len(new_possible_annotations) % 100 == 0:
                            with open('out/debug_logs.txt', 'a') as debug_logs:
                                debug_logs.write(f': {len(new_possible_annotations)} ')
                        if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                            break
                    if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                        break

                    # Extending left
                    extended_span = copy.deepcopy(silver_span)
                    while extended_span.start > left_border:
                        # Extending start of the span
                        initial_word_mark = word_marks[extended_span.start]
                        while word_marks[
                            extended_span.start] == initial_word_mark and extended_span.start > left_border:
                            extended_span.start -= 1
                        new_word_mark = word_marks[extended_span.start]
                        while word_marks[
                            extended_span.start] == new_word_mark and extended_span.start > left_border:
                            extended_span.start -= 1
                        if word_marks[extended_span.start] != new_word_mark:
                            extended_span.start += 1
                        new_possible_annotations.append(annotation + [extended_span])
                        # new_possible_annotations.append(annotation + [extended_span.pos_reversed()])
                        extended_span = copy.deepcopy(extended_span)
                        if len(new_possible_annotations) % 100 == 0:
                            with open('out/debug_logs.txt', 'a') as debug_logs:
                                debug_logs.write(f': {len(new_possible_annotations)} ')
                        if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                            break
                    if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                        break

                    # Extending right
                    extended_span = copy.deepcopy(silver_span)
                    while extended_span.end < right_border:
                        # Extending end of the span
                        initial_word_mark = word_marks[extended_span.end]
                        while word_marks[
                            extended_span.end] == initial_word_mark and extended_span.end < right_border:
                            extended_span.end += 1
                        new_possible_annotations.append(annotation + [extended_span])
                        # new_possible_annotations.append(annotation + [extended_span.pos_reversed()])
                        extended_span = copy.deepcopy(extended_span)
                        if len(new_possible_annotations) % 100 == 0:
                            with open('out/debug_logs.txt', 'a') as debug_logs:
                                debug_logs.write(f': {len(new_possible_annotations)} ')
                        if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                            break
                    if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                        break

                    # Shrinking left
                    shrunk_span = copy.deepcopy(silver_span)
                    while shrunk_span.start < shrunk_span.end:
                        # Shrinking start of the span
                        initial_word_mark = word_marks[shrunk_span.start]
                        while word_marks[
                            shrunk_span.start] == initial_word_mark and shrunk_span.start < shrunk_span.end:
                            shrunk_span.start += 1
                        if shrunk_span.length > 0:
                            new_possible_annotations.append(annotation + [shrunk_span])
                            shrunk_span = copy.deepcopy(shrunk_span)
                            if len(new_possible_annotations) % 100 == 0:
                                with open('out/debug_logs.txt', 'a') as debug_logs:
                                    debug_logs.write(f': {len(new_possible_annotations)} ')
                            if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                                break
                    if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                        break

                    # Shrinking right
                    shrunk_span = copy.deepcopy(silver_span)
                    while shrunk_span.end > shrunk_span.start:
                        # Shrinking end of the span
                        initial_word_mark = word_marks[shrunk_span.end]
                        while word_marks[
                            shrunk_span.end] == initial_word_mark and shrunk_span.start < shrunk_span.end:
                            shrunk_span.end -= 1
                        new_word_mark = word_marks[shrunk_span.end]
                        while word_marks[shrunk_span.end] == new_word_mark and shrunk_span.start < shrunk_span.end:
                            shrunk_span.end -= 1
                        if shrunk_span.start < shrunk_span.end:
                            shrunk_span.end += 1
                        if shrunk_span.length > 0:
                            new_possible_annotations.append(annotation + [shrunk_span])
                            shrunk_span = copy.deepcopy(shrunk_span)
                            if len(new_possible_annotations) % 100 == 0:
                                with open('out/debug_logs.txt', 'a') as debug_logs:
                                    debug_logs.write(f': {len(new_possible_annotations)} ')
                        if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                            break
                    if len(new_possible_annotations) > self.max_possible_fake_anno_num:
                        break

                possible_annotations = new_possible_annotations

        # Remove duplicates
        possible_annotations = list(set([tuple(annotation) for annotation in possible_annotations]))

        with open('out/debug_logs.txt', 'a') as debug_logs:
            debug_logs.write(f' : {len(possible_annotations)}\n')

        possible_fake_tags: list[list[str]] = []
        for annotation in possible_annotations:
            tags: list[str] = ['O' for _ in characters]
            for span in annotation:
                for i in range(span.start, span.end):
                    tags[i] = f'B-{span.pos}' if i == span.start else f'I-{span.pos}'
            if tags == ['O' for _ in characters]:
                continue
            possible_fake_tags.append(tags)

        return possible_fake_tags

    def get_annotation_score(self, worker_id: int, sentence_ids: Union[List[int], int]) \
            -> Tuple[float, List[List[str]], dict[int, int], list[list[str]], list[bool]]:
        """
        Calculate the average score of the worker's annotations on the given sentence set.
        """
        if isinstance(sentence_ids, int):
            sentence_ids = [sentence_ids]

        worker_tags = []
        is_fake_annotations = []
        sent2fake_annotation_num = {sentence_id: 0 for sentence_id in sentence_ids}
        for sentence_id in sentence_ids:
            if worker_id in self.sent2all_tags[sentence_id].keys() and not self.only_fake_annotations:
                worker_tags.append(self.sent2all_tags[sentence_id][worker_id])
                is_fake_annotations.append(False)
            else:
                fake_annotation, *_ = self.get_fake_annotation(sentence_id, worker_id)
                worker_tags.append(fake_annotation)
                is_fake_annotations.append(True)
                sent2fake_annotation_num[sentence_id] += 1

        silver_tags = [self.sent2silver_tags[sentence_id] for sentence_id in sentence_ids]

        if self.metrics_type == 'pearson':
            score = self.metrics.pearson_corr_coef(silver_tags, worker_tags)
        else:
            score = self.metrics.f1_score(silver_tags, worker_tags)

        return score, worker_tags, sent2fake_annotation_num, silver_tags, is_fake_annotations

    def get_unannotated_sentence_ids(self, worker_id: int, num: int = 1) -> List[int]:
        """
        Return the first num sentence ids in worker2unannotated_sents
        """
        return self.worker2unannotated_sents[worker_id][:num]

    def update_unannotated_sentences(self, annotating_worker_id: int, sentence_ids: List[int]) -> NoReturn:
        """
        Remove sentence_id from workers' unannotated_sents.
        """
        for sentence_id in sentence_ids:
            if self.allow_fake_annotations:
                worker_ids = self.worker_ids
            else:
                worker_ids = self.sent2workers[sentence_id]
            for worker_id in worker_ids:
                if sentence_id in self.worker2unannotated_sents[worker_id]:
                    self.worker2unannotated_sents[worker_id].remove(sentence_id)
                # We don't want one worker annotate the same sentence twice.
                if worker_id == annotating_worker_id:
                    # self.worker2unannotated_sents[worker_id] = [
                    #     existing_id
                    #     for existing_id in self.worker2unannotated_sents[worker_id]
                    #     if existing_id != sentence_id
                    # ]
                    # The following implementation is O(N^2) which costs too much time thus deprecated.
                    while sentence_id in self.worker2unannotated_sents[worker_id]:
                        self.worker2unannotated_sents[worker_id].remove(sentence_id)

    def restore_unannotated_sentences(self):
        self.worker2unannotated_sents: dict[int, list[int]] = {}
        for sent, worker_ids in self.sent2workers.items():
            for worker_id in worker_ids:
                if worker_id not in self.worker2unannotated_sents.keys():
                    self.worker2unannotated_sents[worker_id] = []
                self.worker2unannotated_sents[worker_id].append(sent)
        # If one sentence could have multiple annotations, duplicate workers' unannotated_sents
        self.worker2unannotated_sents = {
            worker: sents * self.annotation_num_per_sentence
            for worker, sents in self.worker2unannotated_sents.items()
        }

    def get_all_annotations_report(self) -> Tuple[float, float, float]:
        """
        A simple classification report of all annotations.
        metrics_level: span / token
        """
        silver_annotations = []
        worker_annotations = []
        for sentence_id, silver_tags in self.sent2silver_tags.items():
            for worker_tags in self.sent2all_tags[sentence_id].values():
                silver_annotations.append(silver_tags)
                worker_annotations.append(worker_tags)

        precision = self.metrics.precision_score(silver_annotations, worker_annotations)
        recall = self.metrics.recall_score(silver_annotations, worker_annotations)
        f1 = self.metrics.f1_score(silver_annotations, worker_annotations)

        return precision, recall, f1

    def export_selections(self, workers: List) -> NoReturn:
        sent2selected_workers: Dict[int, List[int]] = {}
        for worker in workers:
            for sentence_id in worker.annotated_sentence_ids:
                if sentence_id in sent2selected_workers.keys():
                    sent2selected_workers[sentence_id].append(worker.id)
                else:
                    sent2selected_workers[sentence_id] = [worker.id]

        selected_data = []
        for sentence in self.data:
            selected_sentence = {
                'id': sentence['id'],
                'text': sentence['text'],
                'annotations': [],
                'reviewer': 1,
                'bestUsers': []
            }
            for annotation in sentence['annotations']:
                if annotation['user'] in sent2selected_workers[sentence['id']]:
                    selected_sentence['annotations'].append(annotation)
            selected_data.append(selected_sentence)

        with open('out/train-online-cucb.json', 'w') as out:
            out.write('[')
            for index, line in enumerate(selected_data):
                out.write(json.dumps(line, ensure_ascii=False))
                if index != len(selected_data) - 1:
                    out.write(',\n')
                else:
                    out.write(']')

    def get_fleiss_kappa_stats(self):
        sent_len2kappas: dict[int, list[float]] = defaultdict(list)
        for sent_id, worker_id2tags in self.sent2all_tags.items():
            worker_tags = [tags for worker_id, tags in worker_id2tags.items()]
            if len(worker_tags) <= 1:
                continue
            kappa = self.metrics.fleiss_kappa(worker_tags)
            sent_len2kappas[len(list(worker_id2tags.values())[0])].append(kappa)
        return sent_len2kappas

    def get_sent2fleiss_kappa(self):
        sent_id2fleiss_kappa = defaultdict(float)
        for sent_id, worker_id2tags in self.sent2all_tags.items():
            worker_tags = [tags for _, tags in worker_id2tags.items()]
            if len(worker_tags) <= 1:
                continue
            sent_id2fleiss_kappa[sent_id] = utils.metrics.fleiss_kappa(worker_tags)
        return dict(sorted(sent_id2fleiss_kappa.items(), key=lambda item: item[1], reverse=True))

    def get_sent2mean_f1(self):
        sent2mean_f1: dict[int, float] = {}
        for sent, workers in self.sent2workers.items():
            worker_annotations = []
            silver_annotations = []
            for worker in workers:
                worker_annotations.append(self.sent2all_tags[sent][worker])
                silver_annotations.append(self.sent2silver_tags[sent])
            sent2mean_f1[sent] = self.metrics.f1_score(silver_annotations, worker_annotations)
        return dict(sorted(sent2mean_f1.items(), key=lambda item: item[1], reverse=True))

    def get_sent2cohens_kappa(self):
        sent_id2cohens_kappa = defaultdict(float)
        for sent_id, worker_id2tags in self.sent2all_tags.items():
            worker_tags = [tags for _, tags in worker_id2tags.items()]
            if len(worker_tags) <= 1:
                continue
            cohens_kappas = []
            for i, tags1 in enumerate(worker_tags):
                for j, tags2 in enumerate(worker_tags):
                    if i <= j:
                        continue
                    cohens_kappas.append(self.metrics.cohens_kappa([tags1, tags2]))
            sent_id2cohens_kappa[sent_id] = float(np.mean(cohens_kappas))
        return dict(sorted(sent_id2cohens_kappa.items(), key=lambda item: item[1], reverse=True))

    def get_sent2same_annotation_num(self):
        sent2same_annotation_num: dict[int, int] = defaultdict(int)
        sent2is_same_selected: dict[int, bool] = defaultdict(lambda: False)
        for sent, worker2tags in self.sent2all_tags.items():
            for i, (worker1, tags1) in enumerate(worker2tags.items()):
                max_same_num = 0
                for j, (worker2, tags2) in enumerate(worker2tags.items()):
                    if tags1 == tags2:
                        max_same_num += 1
                        if i == j and self.sent2silver_tags[sent] == tags1 and max_same_num > 1:
                            sent2is_same_selected[sent] = True
                if max_same_num > sent2same_annotation_num[sent]:
                    sent2same_annotation_num[sent] = max_same_num
        return dict(sorted(sent2same_annotation_num.items(), key=lambda item: item[1], reverse=True)), \
               sent2is_same_selected


def run_cache_annotations(metrics_type: str):
    utils = Utils(annotation_num_per_sentence=4,
                  allow_fake_annotations=True,
                  only_fake_annotations=False,
                  metrics_type=metrics_type,
                  acceptable_fake_anno_error=3000000)
    utils.cache_annotation_matrix()


if __name__ == '__main__':
    # Fleiss Kappa
    utils = Utils(annotation_num_per_sentence=4,
                  allow_fake_annotations=True,
                  only_fake_annotations=False,
                  # metrics_type='span_proportional',
                  metrics_type='span_exact',
                  acceptable_fake_anno_error=3000000)

    # print(np.mean(list(utils.worker2mean_f1.values())))
    # print(np.mean(list(utils.worker2mean_recall.values())))
    # print(np.mean(list(utils.worker2mean_precision.values())))

    utils.cache_annotation_matrix()
    #
    # for key, value in dict(sorted(utils.worker2mean_exact_f1.items(), key=lambda item: item[1], reverse=True)).items():
    #     print(f'{key}: {value}')

    # lengths = []
    # less_than_10_counter = 0
    #
    # span_nums = []
    # lt10_span_nums = []
    #
    # total_span_lengths = []
    # lt10_total_span_lengths = []
    #
    # anno_nums = []
    #
    # for sent_id, silver_tags in utils.sent2silver_tags.items():
    #     # Sent length
    #     lengths.append(len(silver_tags))
    #     if len(silver_tags) <= 10:
    #         less_than_10_counter += 1
    #     # Span num
    #     span_num = sum(map(lambda tag: 1 if 'B-' in tag else 0, silver_tags))
    #     span_nums.append(span_num)
    #     if len(silver_tags) <= 10:
    #         lt10_span_nums.append(span_num)
    #     # Total span length
    #     total_span_length = sum(map(lambda tag: 1 if '-' in tag else 0, silver_tags))
    #     total_span_lengths.append(total_span_length)
    #     if len(silver_tags) <= 10:
    #         lt10_total_span_lengths.append(total_span_length)
    #     # Annotation num per sentence
    #     anno_nums.append(len(utils.sent2all_tags[sent_id]))
    #
    # print(f'sent length: {max(lengths)} - {np.mean(lengths):.02f} - {min(lengths)}')
    # print(f'less than ten %: {less_than_10_counter} / {len(lengths)} = {less_than_10_counter / len(lengths):.02%}')
    # print(f'span # per sent: {max(span_nums)} - {np.mean(span_nums):.02f} - {min(span_nums)}')
    # print(f'span # per lt10 sent: {max(lt10_span_nums)} - {np.mean(lt10_span_nums):.02f} - {min(lt10_span_nums)}')
    # print(f'total span length # per sent: {max(total_span_lengths)} - {np.mean(total_span_lengths):.02f} - {min(total_span_lengths)}')
    # print(f'total span length # per lt10 sent: {max(lt10_total_span_lengths)} - {np.mean(lt10_total_span_lengths):.02f} - {min(lt10_total_span_lengths)}')
    # print(f'anno # per sent: {max(anno_nums)} - {np.mean(anno_nums):.02f} - {min(anno_nums)}')
    #
    # sent_nums = list(map(lambda sent_ids: len(sent_ids), list(utils.worker2sents.values())))
    # print(f'anno # per worker: {max(sent_nums)} - {np.mean(sent_nums):.02f} - {min(sent_nums)}')





