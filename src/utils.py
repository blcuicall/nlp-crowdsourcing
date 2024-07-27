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
                 use_gold_expert=False,
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

        self.use_gold_expert = use_gold_expert

        self.cache_filename = 'out/annotation_matrix.oei.silver'
        if self.use_gold_expert:
            self.cache_filename = self.cache_filename.replace('silver', 'gold')
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
            'B-POS': 1,
            'I-POS': 2,
            'B-NEG': 3,
            'I-NEG': 4,
        }

        with open('data/train.json', 'r') as data_file:
            self.data = json.load(data_file)

        if self.use_gold_expert:
            with open('data/train-max.json', 'r') as gold_file:
                self.gold_data = json.load(gold_file)

        # Drop invalid annotations
        for sentence in self.data:
            filtered_annotations = []
            for annotation in sentence['annotations']:
                if not (annotation['start_offset'] == annotation['end_offset'] == -1):
                    filtered_annotations.append(annotation)
            sentence['annotations'] = filtered_annotations

        # Some given bestUser has no annotation on the sentence.
        # Replace them with a user who has annotation on the sentence.
        for sentence in self.data:
            if not sentence['bestUsers']:
                continue
            best_users = []
            for best_user in sentence['bestUsers']:
                for annotation in sentence['annotations']:
                    if best_user == annotation['user']:
                        best_users.append(best_user)
                        break
            sentence['bestUsers'] = best_users

        # Load memory to keep experiments consistent.
        if memory:
            self.data = memory['data']

        # Sentence split into characters
        self.sentences: dict[int, list[str]] = {}
        for sentence in self.data:
            self.sentences[sentence['id']] = list(sentence['text'])

        # Sentence split into words
        self.sent2words: dict[int, list[str]] = {
            sentence['id']: list(jieba.cut(sentence['text']))
            for sentence in self.data
        }

        # Silver annotation tags of each sentence
        # Example:
        # sent2silver_tags = {
        #     10964: ['O', 'O', 'B-POS', 'I-POS'],
        #     10943: ['B-POS', 'I-POS', 'O', 'O'],
        # }
        self.sent2silver_tags: dict[int, list[str]] = {}
        for sentence in self.data:
            if sentence['bestUsers']:
                best_user = random.choice(sentence['bestUsers'])
            else:
                best_user = random.choice([annotation['user'] for annotation in sentence['annotations']])
            self.sent2silver_tags[sentence['id']] = ['O' for _ in range(len(sentence['text']))]
            for annotation in sentence['annotations']:
                if annotation['user'] == best_user:
                    for i in range(len(sentence['text'])):
                        if i == annotation['start_offset']:
                            self.sent2silver_tags[sentence['id']][i] = f'B-{annotation["label"]}'
                        elif annotation['start_offset'] < i < annotation['end_offset']:
                            self.sent2silver_tags[sentence['id']][i] = f'I-{annotation["label"]}'

        # Load memory to keep experiments consistent.
        if memory:
            self.sent2silver_tags = memory['sent2silver_tags']

        if self.use_gold_expert:
            for sentence in self.gold_data:
                if not sentence['annotations']:
                    continue
                self.sent2silver_tags[sentence['id']] = ['O' for _ in range(len(sentence['text']))]
                for annotation in sentence['annotations']:
                    for i in range(len(sentence['text'])):
                        if i == annotation['start_offset']:
                            self.sent2silver_tags[sentence['id']][i] = f'B-{annotation["label"]}'
                        elif annotation['start_offset'] < i < annotation['end_offset']:
                            self.sent2silver_tags[sentence['id']][i] = f'I-{annotation["label"]}'

        # All annotation tags of each sentence
        # Example:
        # sent2all_tags = {
        #     10964: {1: ['O', 'O', 'B-POS', 'I-POS'], 5: ['O', 'O', 'B-NEG', 'I-NEG']},
        #     4530: {45: ['B-POS', 'I-POS', 'O', 'O'], 33: ['O', 'O', 'B-NEG', 'I-NEG']},
        # }
        self.sent2all_tags: dict[int, dict[int, list[str]]] = {}
        for sentence in self.data:
            self.sent2all_tags[sentence['id']] = {}
            for annotation in sentence['annotations']:
                if annotation['user'] not in self.sent2all_tags[sentence['id']].keys():
                    self.sent2all_tags[sentence['id']][annotation['user']] = ['O' for _ in range(len(sentence['text']))]
                for i in range(len(sentence['text'])):
                    if i == annotation['start_offset']:
                        self.sent2all_tags[sentence['id']][annotation['user']][i] = f'B-{annotation["label"]}'
                    elif annotation['start_offset'] < i < annotation['end_offset']:
                        self.sent2all_tags[sentence['id']][annotation['user']][i] = f'I-{annotation["label"]}'

        # Find out which workers annotated each sentence
        self.sent2workers: dict[int, list[int]] = {}
        for sent in self.data:
            self.sent2workers[sent['id']] = list(set([anno['user'] for anno in sent['annotations']]))

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
            self.worker2mean_prop_precision[worker] = self.metrics.precision_score(silver_annotations,
                                                                                   worker_annotations,
                                                                                   metrics_type='span_proportional')
            self.worker2mean_exact_precision[worker] = self.metrics.precision_score(silver_annotations,
                                                                                    worker_annotations,
                                                                                    metrics_type='span_exact')

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
            self.worker2mean_prop_recall[worker] = self.metrics.recall_score(silver_annotations, worker_annotations,
                                                                             metrics_type='span_proportional')
            self.worker2mean_exact_recall[worker] = self.metrics.recall_score(silver_annotations, worker_annotations,
                                                                              metrics_type='span_exact')

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
            self.worker2mean_prop_f1[worker] = self.metrics.f1_score(silver_annotations, worker_annotations,
                                                                     metrics_type='span_proportional')
            self.worker2mean_exact_f1[worker] = self.metrics.f1_score(silver_annotations, worker_annotations,
                                                                      metrics_type='span_exact')

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
        words = jieba.cut(''.join(characters))
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
                exact_precisions.append(
                    self.metrics.precision_score([silver_annotation], [fake_annotation], metrics_type='span_exact'))
                exact_recalls.append(
                    self.metrics.recall_score([silver_annotation], [fake_annotation], metrics_type='span_exact'))
                exact_f1s.append(
                    self.metrics.f1_score([silver_annotation], [fake_annotation], metrics_type='span_exact'))
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

        # Build annotation span by span.
        # fake_annotation: list[str] = []
        # is_span_correct = True
        # for i, silver_tag in enumerate(self.sent2silver_tags[sentence_id]):
        #     if silver_tag.startswith('B'):
        #         is_span_correct = random.random() < self.worker2mean_f1[worker_id]
        #     if is_span_correct:
        #         # Use expert annotation.
        #         fake_annotation.append(silver_tag)
        #     else:
        #         # Use all zeros.
        #         fake_annotation.append('O')
        #
        #         # Use randomly generated annotation.
        #         # choices = [tag for tag in self.tag2index.keys()]
        #         # if i == 0 or fake_annotation[i - 1] == 'O':
        #         #     choices.remove('I-POS')
        #         #     choices.remove('I-NEG')
        #         # elif fake_annotation[i - 1] == 'B-POS':
        #         #     choices.remove('B-POS')
        #         #     choices.remove('I-NEG')
        #         # elif fake_annotation[i - 1] == 'B-NEG':
        #         #     choices.remove('B-NEG')
        #         #     choices.remove('I-POS')
        #         # elif fake_annotation[i - 1] == 'I-POS':
        #         #     choices.remove('B-POS')
        #         #     choices.remove('I-NEG')
        #         # elif fake_annotation[i - 1] == 'I-NEG':
        #         #     choices.remove('B-NEG')
        #         #     choices.remove('I-POS')
        #         #
        #         # fake_annotation.append(random.choice(choices))
        #
        # self.fake_precision_errors.append(abs(self.metrics.precision_score([self.sent2silver_tags[sentence_id]], [fake_annotation]) - self.worker2mean_precision[worker_id]))
        # self.fake_recall_errors.append(abs(self.metrics.recall_score([self.sent2silver_tags[sentence_id]], [fake_annotation]) - self.worker2mean_recall[worker_id]))
        # self.fake_f1_errors.append(abs(self.metrics.f1_score([self.sent2silver_tags[sentence_id]], [fake_annotation]) - self.worker2mean_f1[worker_id]))
        #
        # return fake_annotation

        # # Return expert annotation.
        # if random.random() < self.worker2mean_f1[worker_id]:
        #     return [tag for tag in self.sent2silver_tags[sentence_id]]
        #
        # # Return randomly generated annotation.
        # fake_annotation = []
        # for i, _ in enumerate(self.sent2silver_tags[sentence_id]):
        #     choices = [tag for tag in self.tag2index.keys()]
        #     if i == 0 or fake_annotation[i - 1] == 'O':
        #         choices.remove('I-POS')
        #         choices.remove('I-NEG')
        #     elif fake_annotation[i - 1] == 'B-POS':
        #         choices.remove('B-POS')
        #         choices.remove('I-NEG')
        #     elif fake_annotation[i - 1] == 'B-NEG':
        #         choices.remove('B-NEG')
        #         choices.remove('I-POS')
        #     elif fake_annotation[i - 1] == 'I-POS':
        #         choices.remove('B-POS')
        #         choices.remove('I-NEG')
        #     elif fake_annotation[i - 1] == 'I-NEG':
        #         choices.remove('B-NEG')
        #         choices.remove('I-POS')
        #
        #     fake_annotation.append(random.choice(choices))
        # return fake_annotation

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
                scores: dict[str, float] = self.metrics.span_proportional_scores([self.sent2silver_tags[sent_id]],
                                                                                 [tags])
                worker_id2fake_prop_precisions[worker_id].append(scores['precision'])
                worker_id2fake_prop_recalls[worker_id].append(scores['recall'])
                worker_id2fake_prop_f1s[worker_id].append(scores['F1'])
                worker_id2fake_exact_precisions[worker_id].append(
                    self.metrics.precision_score([self.sent2silver_tags[sent_id]], [tags], metrics_type='span_exact'))
                worker_id2fake_exact_recalls[worker_id].append(
                    self.metrics.recall_score([self.sent2silver_tags[sent_id]], [tags], metrics_type='span_exact'))
                worker_id2fake_exact_f1s[worker_id].append(
                    self.metrics.f1_score([self.sent2silver_tags[sent_id]], [tags], metrics_type='span_exact'))

        worker_id2real_f1 = dict(sorted(self.worker2mean_f1.items(), key=lambda item: item[1], reverse=True))

        with open(self.cache_filename.replace('.json', '.report.csv'), 'w') as out:
            out.write(f', Prop, Prop, Prop, Prop, Prop, Prop, Exact, Exact, Exact, Exact, Exact, Exact\n')
            out.write(
                f'Worker ID, Real P, Fake P, Real R, Fake R, Real F1, Fake F1, Real P, Fake P, Real R, Fake R, Real F1, Fake F1\n')
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

    def get_possible_fake_annotations_random_gen(self, characters: list[str], words: list[str]):
        # Preprocessing
        # Mark characters with corresponding words.
        word_marks: list[int] = []
        for i, word in enumerate(words):
            for _ in list(word):
                word_marks.append(i)
        word_marks.append(-1)

        with open('out/debug_logs.txt', 'a') as debug_logs:
            debug_logs.write(f'{"".join(characters)}; len={len(characters)}; ')

        possible_annotations: list[list[str]] = []

        for i, character in enumerate(characters):
            if i == 0:
                choices = [tag for tag in self.tag2index.keys()]
                choices.remove('I-POS')
                choices.remove('I-NEG')
                for choice in choices:
                    possible_annotations.append([choice])
            else:
                last_possible_annotations = possible_annotations
                possible_annotations = []
                for fake_annotation in last_possible_annotations:
                    # Check if is in the same word.
                    if word_marks[i] == word_marks[i - 1]:
                        if fake_annotation[i - 1] == 'B-POS':
                            possible_annotations.append(fake_annotation + ['I-POS'])
                        elif fake_annotation[i - 1] == 'B-NEG':
                            possible_annotations.append(fake_annotation + ['I-NEG'])
                        else:
                            possible_annotations.append(fake_annotation + [fake_annotation[-1]])
                    else:
                        choices = [tag for tag in self.tag2index.keys()]
                        if fake_annotation[i - 1] == 'O':
                            choices.remove('I-POS')
                            choices.remove('I-NEG')
                        elif fake_annotation[i - 1] == 'B-POS':
                            choices.remove('B-POS')
                            choices.remove('I-NEG')
                        elif fake_annotation[i - 1] == 'B-NEG':
                            choices.remove('B-NEG')
                            choices.remove('I-POS')
                        elif fake_annotation[i - 1] == 'I-POS':
                            choices.remove('B-POS')
                            choices.remove('I-NEG')
                        elif fake_annotation[i - 1] == 'I-NEG':
                            choices.remove('B-NEG')
                            choices.remove('I-POS')
                        for choice in choices:
                            if len(possible_annotations) > self.max_possible_fake_anno_num:
                                break
                            possible_annotations.append(fake_annotation + [choice])
            with open('out/debug_logs.txt', 'a') as debug_logs:
                debug_logs.write(f'{i}:{len(possible_annotations)}; ')

        with open('out/debug_logs.txt', 'a') as debug_logs:
            debug_logs.write('\n')
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

            def pos_reversed(self):
                reversed_span = copy.deepcopy(self)
                reversed_span.pos = 'NEG' if reversed_span.pos == 'POS' else 'POS'
                return reversed_span

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
                possible_annotations.append([silver_span.pos_reversed()])

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
                    new_possible_annotations.append(annotation + [silver_span.pos_reversed()])

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

    def get_unannotated_sentence_ids(self, worker_id: int, num: int) -> List[int]:
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

    def get_worker2avg_span_length(self):
        worker2lengths: dict[int, list[float]] = defaultdict(list)

        class Span:
            def __init__(self, start: Optional[int] = None, end: Optional[int] = None, pos: Optional[str] = None):
                self.start = start  # Span includes the start position
                self.end = end  # Span does not include the end position
                self.pos = pos

            @property
            def length(self):
                return self.end - self.start

        for sent, worker2tags in self.sent2all_tags.items():
            for worker, tags in worker2tags.items():
                spans = []
                is_in_span = False
                current_span = Span()
                for i, tag in enumerate(tags):
                    if is_in_span:
                        if tag.startswith('I'):
                            continue
                        current_span.end = i
                        spans.append(current_span)
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
                    current_span.end = len(tags)
                    spans.append(current_span)
                for span in spans:
                    worker2lengths[worker].append(span.length)

        return {worker: np.mean(lengths) for worker, lengths in worker2lengths.items()}

    def get_sents_and_workers_and_avg_span_lengths_and_f1s(self):

        class Span:
            def __init__(self, start: Optional[int] = None, end: Optional[int] = None, pos: Optional[str] = None):
                self.start = start  # Span includes the start position
                self.end = end  # Span does not include the end position
                self.pos = pos

            @property
            def length(self):
                return self.end - self.start

        sents = []
        workers = []
        span_lengths = []
        f1s = []

        crowd_spans: list[list[Span]] = []
        sent2expert_spans: dict[int, list[Span]] = {}
        expert_spans: list[list[Span]] = []

        for sent, tags in self.sent2silver_tags.items():
            spans = []
            is_in_span = False
            current_span = Span()
            for i, tag in enumerate(tags):
                if is_in_span:
                    if tag.startswith('I'):
                        continue
                    current_span.end = i
                    spans.append(current_span)
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
                current_span.end = len(tags)
                spans.append(current_span)
            sent2expert_spans[sent] = spans

        for sent, worker2tags in self.sent2all_tags.items():
            for worker, tags in worker2tags.items():
                spans = []
                is_in_span = False
                current_span = Span()
                for i, tag in enumerate(tags):
                    if is_in_span:
                        if tag.startswith('I'):
                            continue
                        current_span.end = i
                        spans.append(current_span)
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
                    current_span.end = len(tags)
                    spans.append(current_span)
                sents.append(sent)
                workers.append(worker)
                span_lengths.append(np.mean([span.length for span in spans]))
                f1s.append(self.metrics.f1_score([self.sent2silver_tags[sent]], [tags]))

                crowd_spans.append(spans)
                expert_spans.append(sent2expert_spans[sent])

        # plt.scatter(span_lengths, f1s)
        # plt.show()

        long_span_error_num = 0
        respective_long_span_error_num = 0
        shift_error_num = 0
        shrink_error_num = 0
        correct_num = 0
        error_threshold = 0.98
        for i, (span_length, f1, current_expert_spans, current_crowd_spans) in enumerate(zip(span_lengths, f1s, expert_spans, crowd_spans)):
            if f1 > error_threshold:
                correct_num += 1
                # print(f'good, sent: {sents[i]}, worker: {workers[i]}\n'
                #       f'join: {"".join(self.sentences[sents[i]])}\n'
                #       f'text: {self.sentences[sents[i]]}\n'
                #       f'exp tags: {self.sent2silver_tags[sents[i]]}\n'
                #       f'crd tags: {self.sent2all_tags[sents[i]][workers[i]]}')
                continue

            if span_length >= 15:
                long_span_error_num += 1
                # print(f'bad, sent: {sents[i]}, worker: {workers[i]}\n'
                #       f'join: {"".join(self.sentences[sents[i]])}\n'
                #       f'text: {self.sentences[sents[i]]}\n'
                #       f'exp tags: {self.sent2silver_tags[sents[i]]}\n'
                #       f'crd tags: {self.sent2all_tags[sents[i]][workers[i]]}')

            respective_span_length = span_length / len(self.sent2silver_tags[sents[i]])
            if respective_span_length >= 0.5:
                respective_long_span_error_num += 1

            is_shift_found = False
            for expert_span in current_expert_spans:
                if is_shift_found:
                    break
                for crowd_span in current_crowd_spans:
                    if expert_span.length == crowd_span.length and expert_span.start != crowd_span.start:
                        print(f'shift, sent: {sents[i]}, worker: {workers[i]}\n'
                              f'join: {"".join(self.sentences[sents[i]])}\n'
                              f'text: {self.sentences[sents[i]]}\n'
                              f'exp tags: {self.sent2silver_tags[sents[i]]}\n'
                              f'crd tags: {self.sent2all_tags[sents[i]][workers[i]]}')
                        is_shift_found = True
                        break


        print(f'fully correct: {correct_num} / {len(f1s)} = {correct_num / len(f1s):.02%}')
        error_num = len(f1s) - correct_num
        print(f'long span error: {long_span_error_num} / {len(f1s)} = {long_span_error_num / len(f1s):.02%} '
              f'{long_span_error_num} / {error_num} = {long_span_error_num / error_num:.02%}')
        print(f'respective long span error: {respective_long_span_error_num} / {len(f1s)} = '
              f'{respective_long_span_error_num / len(f1s):.02%} '
              f'{respective_long_span_error_num} / {error_num} = {respective_long_span_error_num / error_num:.02%}')

        return


def run_cache_annotations(use_gold_expert: bool, metrics_type: str):
    utils = Utils(annotation_num_per_sentence=4,
                  allow_fake_annotations=True,
                  only_fake_annotations=False,
                  metrics_type=metrics_type,
                  use_gold_expert=use_gold_expert,
                  acceptable_fake_anno_error=3000000)
    utils.cache_annotation_matrix()


if __name__ == '__main__':
    # Fleiss Kappa
    utils = Utils(annotation_num_per_sentence=4,
                  allow_fake_annotations=True,
                  only_fake_annotations=False,
                  # metrics_type='span_proportional',
                  metrics_type='span_exact',
                  use_gold_expert=True,
                  acceptable_fake_anno_error=3000000)

    utils.cache_annotation_matrix()

    # worker2original_f1 = dict(sorted(utils.worker2mean_exact_f1.items(), key=lambda item: item[1], reverse=True))
    # aug_f1s = []
    # for worker_id, original_f1 in worker2original_f1.items():
    #     num = len(utils.worker2sents[worker_id]) #original_sent_num
    #     aug_f1 = (original_f1 * num + random.random() * (len(utils.sentences) - num)) / len(utils.sentences)
    #     aug_f1s.append(aug_f1)
    #     print(f'{worker_id}: {original_f1:.02%}: {aug_f1:.02%}')
    # print(aug_f1s)

    # How long are expert spans?

    # How long are crowd spans?
    # worker2length = dict(sorted(utils.get_worker2avg_span_length().items(), reverse=True, key=lambda item: item[1]))
    # print(worker2length)
    # lengths: list[float] = [float(value) for value in worker2length.values()]
    # f1s: list[float] = [float(value) for value in utils.worker2mean_exact_f1.values()]
    # plt.scatter(lengths, f1s)
    # plt.show()
    # print(f'avg: {np.mean(lengths)}')

    # utils.get_sents_and_workers_and_avg_span_lengths_and_f1s()
