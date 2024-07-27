import json
import random
from collections import OrderedDict, defaultdict
from functools import reduce, partial
from typing import Union, NoReturn, Tuple, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from prettytable import PrettyTable
from tqdm import tqdm, trange

from utils import Utils


class Worker:
    """
    A worker in the annotating system, corresponding to the arm in the CMAB algorithm.
    """

    def __init__(self, id: int, utils: Utils):
        self.id = id
        self.f1s = []
        self.annotation_num = 0
        self.annotated_sentence_ids = []
        self.utils = utils

    @property
    def average_f1(self):
        # This costs a lot of time, to be optimised.
        return np.mean(self.f1s)

    def annotate(self, sentence_ids: Union[list[int], int], update_f1: bool = True, split_spans: bool = False) \
            -> Tuple[list[list[str]], float, dict[int, int], list[bool]]:
        """
        Annotate new sentences, and return those annotations.
        """
        # Get score with util methods
        f1, annotations, sent2fake_annotation_num, silver_tags, is_fake_annotations = self.utils.get_annotation_score(
            self.id,
            sentence_ids)
        self.annotated_sentence_ids.extend(sentence_ids)

        if update_f1:
            self.f1s.append(f1)

        if split_spans:
            for sentence_silver_tags in silver_tags:
                span_num = 0
                for tag in sentence_silver_tags:
                    if 'B-' in tag:
                        span_num += 1
                self.annotation_num += span_num
        else:
            self.annotation_num += len(sentence_ids)
        self.utils.update_unannotated_sentences(self.id, sentence_ids)
        return annotations, f1, sent2fake_annotation_num, is_fake_annotations

    def __repr__(self):
        return f'{self.id}'


class Manager:
    """
    The annotating system manager, corresponding to the bandit player in the CMAB algorithm.
    """

    def __init__(self, workers: list[Worker], utils: Utils, selected_worker_num: int = 20, ucb_scale: float = 1.0,
                 allow_exhaustion: bool = True, evaluation_interval: int = 10, split_spans: bool = False):
        self.workers = workers
        self.selected_worker_num = selected_worker_num
        self.worker_num = len(self.workers)

        self.exhausted_workers = []
        self.allow_exhaustion = allow_exhaustion

        self.evaluation_interval = evaluation_interval
        self.time_since_last_evaluation = evaluation_interval  # evaluate in the first round
        self.last_selection: list[Worker] = []

        self.t = 0  # Time step
        self.ucb_scale = ucb_scale

        self.annotations: dict[int, list[list[str]]] = defaultdict(list)
        self.annotation_f1s: dict[int, list[float]] = defaultdict(list)
        self.annotation_workers: dict[int, list[Worker]] = defaultdict(list)
        self.sent2eval_type: dict[int, int] = {} # 1 for expert, 2 for mv

        self.fake_annotations: dict[int, list[list[str]]] = {}

        self.selection_history = []
        self.regret_history = []
        self.utils = utils

        self.sum_empirical_best_f1s = sum(
            sorted(self.utils.worker2mean_f1.values(), reverse=True)[:selected_worker_num])

        self.split_spans = split_spans

        self.sent2fake_annotation_num: dict[int, int] = {
            sentence_id: 0 for sentence_id in self.utils.sent2workers.keys()
        }

    @property
    def run_name(self):
        return f'{self.__class__.__name__}' \
               f'-{self.utils.metrics_type}' \
               f'-aps={self.utils.annotation_num_per_sentence}' \
               f'-wn={self.selected_worker_num}'

    def initialize(self) -> NoReturn:
        """
        Before normal rounds, let each worker annotate once for reference in later selections.
        """
        for worker in self.workers:
            self.t += 1
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            annotations, f1, sent2fake_annotation_num, _ = worker.annotate(sentence_ids, split_spans=self.split_spans)
            for sentence_id, annotation in zip(sentence_ids, annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.sent2eval_type[sentence_id] = 1
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]

            self.selection_history.append(worker.id)

    def run_one_step(self) -> NoReturn:
        """
        Select the best workers according to their current average F-1 scores and let them annotate sentences.
        """
        self.t += 1

        is_eval_step = self.time_since_last_evaluation == self.evaluation_interval

        if is_eval_step:
            # Select the best workers with consideration of upper confidence bounds.
            worker2score: dict[Worker, float] = {
                worker: worker.average_f1 + self.ucb_scale * np.sqrt(3 * np.log(self.t) / (2 * worker.annotation_num))
                for worker in self.workers if worker not in self.exhausted_workers
            }

            selected_workers: list[Worker] = [
                worker
                for worker, _ in
                sorted(worker2score.items(), key=lambda item: item[1], reverse=True)[:self.selected_worker_num]
            ]

            self.last_selection = selected_workers
            self.time_since_last_evaluation = 0
        else:
            # Use last selection.
            selected_workers = self.last_selection

            self.time_since_last_evaluation += 1

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)

            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            sent_anno_num += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=is_eval_step,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)
                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)

    def run(self, num_steps: int) -> NoReturn:
        """
        Run the whole process.
        Args:
            num_steps: Number of time steps to take.
        """
        assert num_steps >= self.worker_num, \
            f'Given num_steps {num_steps} are less than arm_num {self.worker_num}. ' \
            f'Algorithm would end during initialization.'
        self.initialize()
        for _ in tqdm(range(num_steps - self.worker_num), desc='Running', position=1, leave=False, ncols=80):
            self.run_one_step()

    def evaluate(self, dump_results: bool = True, metrics_type: Optional[str] = None) \
            -> Tuple[float, float, float, int, float, float, float, float, float]:
        """
        Calculate classification scores and dump results.
        """

        # Calculate overall average F-1 score of annotations
        worker_annotations = []
        silver_annotations = []
        for sentence_id, sentence_annotations in self.annotations.items():
            for annotation in sentence_annotations:
                worker_annotations.append(annotation)
                silver_annotations.append(self.utils.sent2silver_tags[sentence_id])

        # The calculation of these scores cost too much time. About 1 min for time step = 10000 and eval interval = 0.
        precision = self.utils.metrics.precision_score(silver_annotations, worker_annotations,
                                                       metrics_type=metrics_type)
        recall = self.utils.metrics.recall_score(silver_annotations, worker_annotations, metrics_type=metrics_type)
        f1 = self.utils.metrics.f1_score(silver_annotations, worker_annotations, metrics_type=metrics_type)
        if self.utils.metrics_type == 'pearson':
            pcc = self.utils.metrics.pearson_corr_coef(silver_annotations, worker_annotations)
        else:
            pcc = 0

        # Calculate average F-1 score of fake annotations
        fake_annotations = []
        silver_annotations_for_fakes = []
        for sentence_id, sentence_annotations in self.fake_annotations.items():
            for annotation in sentence_annotations:
                fake_annotations.append(annotation)
                silver_annotations_for_fakes.append(self.utils.sent2silver_tags[sentence_id])

        fake_precision = self.utils.metrics.precision_score(silver_annotations_for_fakes, fake_annotations,
                                                            metrics_type=metrics_type)
        fake_recall = self.utils.metrics.recall_score(silver_annotations_for_fakes, fake_annotations,
                                                      metrics_type=metrics_type)
        fake_f1 = self.utils.metrics.f1_score(silver_annotations_for_fakes, fake_annotations, metrics_type=metrics_type)
        if self.utils.metrics_type == 'pearson':
            fake_pcc = self.utils.metrics.pearson_corr_coef(silver_annotations_for_fakes, fake_annotations)
        else:
            fake_pcc = 0

        if dump_results:
            with open('out/statistics.csv', 'w') as out:
                out.write(f'Worker ID, Annotation #, Avg F-1, Exhausted\n')
                for worker in sorted(self.workers, key=lambda worker: worker.annotation_num, reverse=True):
                    out.write(f'{worker.id}, '
                              f'{worker.annotation_num}/{len(self.utils.worker2sents[worker.id])}, '
                              f'{worker.average_f1 * 100:.02f}, '
                              f'{worker in self.exhausted_workers}\n')

            with open('out/selection_log.txt', 'w') as out:
                for selection in self.selection_history:
                    out.write(f'{selection}\n')

        annotation_num = reduce(lambda num1, num2: num1 + num2, map(lambda worker: worker.annotation_num, self.workers))

        return precision, recall, f1, annotation_num, fake_precision, fake_recall, fake_f1, pcc, fake_pcc

    def plot_regret(self, regret_history=None, cumulative=True):
        history = regret_history if regret_history is not None else self.regret_history
        agg_step = 1
        regret_data = pd.DataFrame({
            'time': range(1, len(history) + 1) if cumulative else [(int(i / agg_step) + 1) * agg_step for i in
                                                                   range(1, len(history) + 1)],
            'regret': np.cumsum(history) if cumulative else history
        })
        sns.lineplot(x='time', y='regret', data=regret_data)
        # plt.title(title)
        plt.xlabel('Time step')
        plt.ylabel('Regret')
        plt.savefig(f'out/pics/oei/{self.run_name}.eps', dpi=600)
        plt.show()

        with open(f'out/regret_logs/oei/{self.run_name}.txt', 'w') as out:
            for step in history:
                out.write(f'{step}\n')

    def plot_sent_len2f1_dist(self):
        sent_len2f1s = OrderedDict()
        for sentence_id in self.annotations.keys():
            sent_len = len(self.annotations[sentence_id][0])
            if sent_len not in sent_len2f1s:
                sent_len2f1s[sent_len] = [f1 for f1 in self.annotation_f1s[sentence_id]]
            else:
                sent_len2f1s[sent_len].extend(self.annotation_f1s[sentence_id])
        sent_len2mean_f1 = OrderedDict({sent_len: np.mean(f1s) for sent_len, f1s in sent_len2f1s.items()})
        sent_len2sent_num = {
            sent_len: len(sent_len2f1s[sent_len])
            for sent_len, mean_f1 in sent_len2mean_f1.items()
        }
        sent_len2normalized_sent_num = {
            sent_len: (color - min(sent_len2sent_num.values())) / (
                    max(sent_len2sent_num.values()) - min(sent_len2sent_num.values()))
            for sent_len, color in sent_len2sent_num.items()
        }
        plt.rcParams['figure.figsize'] = (10.0, 5.0)
        lower_color = np.divide([191, 239, 255, 255], 255)
        upper_color = np.divide([24, 116, 205, 255], 255)
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [lower_color, upper_color])
        colors = custom_cmap(list(sent_len2normalized_sent_num.values()))
        for index, (sent_len, mean_f1) in enumerate(sent_len2mean_f1.items()):
            plt.bar(sent_len, mean_f1, color=colors[index])
        plt.colorbar(ScalarMappable(cmap=custom_cmap,
                                    norm=mpl.colors.Normalize(min(sent_len2sent_num.values()),
                                                              max(sent_len2sent_num.values()))),
                     ticks=np.linspace(min(sent_len2sent_num.values()), max(sent_len2sent_num.values()), 10),
                     label='Sentence #')
        plt.xlabel('Sentence Length')
        plt.ylabel('Average F1')
        plt.show()

    def dump_annotations(self, annotations=None, annotation_workers=None, annotation_eval_types=None, filename_suffix=''):
        sent2annotations = annotations if annotations is not None else self.annotations
        sent2workers = annotation_workers if annotation_workers is not None else self.annotation_workers
        sent2eval_type = annotation_eval_types if annotation_eval_types is not None else self.sent2eval_type
        dump_data = []
        for sent_id, worker_tags in sent2annotations.items():
            sent_data = {
                'id': sent_id,
                'text': ''.join(self.utils.sentences[sent_id]),
                'annotations': [],
                'reviewer': sent2eval_type[sent_id],
                'bestUsers': []
            }
            for worker, tags in zip(sent2workers[sent_id], worker_tags):
                span = {
                    "label": None,
                    "start_offset": -1,
                    "end_offset": -1,
                    "user": worker.id
                }
                is_in_span = False
                for i, tag in enumerate(tags):
                    if not is_in_span and tag == 'O':
                        continue
                    elif not is_in_span and tag.startswith('B-'):
                        is_in_span = True
                        span['start_offset'] = i
                        span['end_offset'] = i + 1
                        span['label'] = tag.removeprefix('B-')
                    elif is_in_span and tag.startswith('I-'):
                        span['end_offset'] += 1
                    elif is_in_span and tag == 'O':
                        sent_data['annotations'].append(span)
                        span = {
                            "label": None,
                            "start_offset": -1,
                            "end_offset": -1,
                            "user": worker.id
                        }
                        is_in_span = False
                    elif is_in_span and tag.startswith('B-'):
                        sent_data['annotations'].append(span)
                        span = {
                            "label": tag.removeprefix('B-'),
                            "start_offset": i,
                            "end_offset": i + 1,
                            "user": worker.id
                        }
                    else:
                        raise ValueError(f'Invalid annotation found, sent: {sent_id}, worker: {worker.id}')
            dump_data.append(sent_data)
        with open(f'out/annotations/oei/{self.run_name}{filename_suffix}.json', 'w') as out:
            out.write('[')
            for index, line in enumerate(dump_data):
                out.write(json.dumps(line, ensure_ascii=False))
                if index != len(dump_data) - 1:
                    out.write(',\n')
                else:
                    out.write(']')


class BestManager(Manager):
    """
    Always select the best workers according to the average F-1 on real dataset in each round.
    """

    def run_one_step(self) -> NoReturn:
        self.t += 1

        # Select the best workers
        worker2score: dict[Worker, float] = {
            worker: self.utils.worker2mean_f1[worker.id]
            for worker in self.workers if worker not in self.exhausted_workers
        }

        selected_workers: list[Worker] = [
            worker
            for worker, _ in
            sorted(worker2score.items(), key=lambda item: item[1], reverse=True)[:self.selected_worker_num]
        ]

        self.last_selection = selected_workers
        self.time_since_last_evaluation = 0

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            sent_anno_num += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=False,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)
                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)


class WorstManager(Manager):
    """
    Always select the worst workers according to the average F-1 on real dataset in each round.
    """

    def run_one_step(self) -> NoReturn:
        self.t += 1

        # Select the worst workers
        worker2score: dict[Worker, float] = {
            worker: self.utils.worker2mean_f1[worker.id]
            for worker in self.workers if worker not in self.exhausted_workers
        }

        selected_workers: list[Worker] = [
            worker
            for worker, _ in
            sorted(worker2score.items(), key=lambda item: item[1], reverse=False)[:self.selected_worker_num]
        ]

        self.last_selection = selected_workers
        self.time_since_last_evaluation = 0

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            sent_anno_num += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=False,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)
                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)


class RandomManager(Manager):
    """
    Always select random workers in each round.
    """

    def run_one_step(self) -> NoReturn:
        self.t += 1

        # Select random workers
        random.shuffle(self.workers)
        selected_workers = self.workers[:self.selected_worker_num]

        self.last_selection = selected_workers
        self.time_since_last_evaluation = 0

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            sent_anno_num += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=False,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)
                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)


class MVExpertManager(Manager):
    """
    Use majority voting to decide when to access the expert annotations.
    """

    def __init__(self, workers: list[Worker], utils: Utils, acceptable_mv_char_error_nums: int = 0,
                 acceptable_mv_percentage: float = 0.67, **kwargs):
        super().__init__(workers, utils, **kwargs)
        self.expert_use_num = 0
        self.sent2expert_tag_ratio = {}

        self.sent2is_mv_tags_correct: dict[int, bool] = {}  # Sentence level correct
        self.sent2mv_tags_correct_ratio: dict[int, float] = {}  # Token level correct

        self.confidence_bounds: list[float] = []

        self.acceptable_mv_char_error_nums = acceptable_mv_char_error_nums
        self.acceptable_mv_percentage = acceptable_mv_percentage

    def initialize(self) -> NoReturn:
        """
        Before normal rounds, let each worker annotate once for reference in later selections.
        """
        for worker in self.workers:
            self.t += 1
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            annotations, f1, sent2fake_annotation_num, _ = worker.annotate(sentence_ids, split_spans=self.split_spans)
            for sentence_id, annotation in zip(sentence_ids, annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)
                self.sent2eval_type[sentence_id] = 1
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
            self.selection_history.append(worker.id)

    def run_one_step(self) -> NoReturn:
        """
        Select the best workers according to their current average F-1 scores and let them annotate sentences.
        """
        self.t += 1

        is_eval_step = self.time_since_last_evaluation == self.evaluation_interval

        if is_eval_step:
            # Select the best workers with consideration of upper confidence bounds.
            worker2score: dict[Worker, float] = {
                worker: worker.average_f1 + self.ucb_scale * np.sqrt(3 * np.log(self.t) / (2 * worker.annotation_num))
                for worker in self.workers if worker not in self.exhausted_workers
            }

            # Record confidence bounds
            for worker in self.workers:
                if worker not in self.exhausted_workers:
                    self.confidence_bounds.append(
                        self.ucb_scale * np.sqrt(3 * np.log(self.t) / (2 * worker.annotation_num)))

            selected_workers: list[Worker] = [
                worker
                for worker, _ in
                sorted(worker2score.items(), key=lambda item: item[1], reverse=True)[:self.selected_worker_num]
            ]

            self.last_selection = selected_workers
            self.time_since_last_evaluation = 0
        else:
            # Use last selection.
            selected_workers = self.last_selection

            self.time_since_last_evaluation += 1

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            last_sentence_ids += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=False,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)
                self.sent2eval_type[sentence_id] = 1
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)

                # Check if all annotations on this sentence are done.
                if self.utils.allow_fake_annotations:
                    voter_num = self.utils.annotation_num_per_sentence
                else:
                    voter_num = min(len(self.utils.sent2workers[sentence_id]), self.utils.annotation_num_per_sentence)
                if len(self.annotations[sentence_id]) == voter_num:
                    # Update f1 score records of workers.
                    mv_expert_tags = self.get_mv_expert_tags(sentence_id)
                    if is_eval_step:
                        for worker, annotation in zip(self.annotation_workers[sentence_id],
                                                      self.annotations[sentence_id]):
                            if self.utils.metrics_type == 'pearson':
                                worker.f1s.append(self.utils.metrics.pearson_corr_coef([mv_expert_tags], [annotation]))
                            else:
                                worker.f1s.append(self.utils.metrics.f1_score([mv_expert_tags], [annotation]))

                    # Check if mv_expert_tags is the same as the real expert tags.
                    self.sent2mv_tags_correct_ratio[sentence_id] = 0
                    for mv_tag, expert_tag in zip(mv_expert_tags, self.utils.sent2silver_tags[sentence_id]):
                        if mv_tag == expert_tag:
                            self.sent2mv_tags_correct_ratio[sentence_id] += 1
                    self.sent2is_mv_tags_correct[sentence_id] = self.sent2mv_tags_correct_ratio[sentence_id] == len(
                        mv_expert_tags)
                    self.sent2mv_tags_correct_ratio[sentence_id] /= len(mv_expert_tags)

                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)

    def get_mv_expert_tags(self, sentence_id):
        mv_expert_tags = []
        mv_tags = []
        votes = [{} for _ in self.utils.sent2silver_tags[sentence_id]]
        for annotation in self.annotations[sentence_id]:
            for i, tag in enumerate(annotation):
                if tag not in votes[i].keys():
                    votes[i][tag] = 0
                votes[i][tag] += 1

        # Select tags according to votes.
        if self.utils.allow_fake_annotations:
            voter_num = self.utils.annotation_num_per_sentence
        else:
            voter_num = min(len(self.utils.sent2workers[sentence_id]), self.utils.annotation_num_per_sentence)
        self.sent2expert_tag_ratio[sentence_id] = 0
        for i, vote in enumerate(votes):
            sorted_vote = {k: v for k, v in sorted(vote.items(), key=lambda item: item[1], reverse=True)}
            if list(sorted_vote.items())[0][1] / voter_num >= self.acceptable_mv_percentage and voter_num != 1:
                mv_expert_tags.append(list(sorted_vote.items())[0][0])
            else:
                mv_expert_tags.append(self.utils.sent2silver_tags[sentence_id][i])
                self.sent2expert_tag_ratio[sentence_id] += 1
            mv_tags.append(list(sorted_vote.items())[0][0])
        if self.sent2expert_tag_ratio[sentence_id] <= self.acceptable_mv_char_error_nums:
            self.sent2expert_tag_ratio[sentence_id] = 0
            mv_expert_tags = mv_tags
        if self.sent2expert_tag_ratio[sentence_id] > 0:
            self.expert_use_num += 1
        self.sent2expert_tag_ratio[sentence_id] /= len(self.utils.sent2silver_tags[sentence_id])

        # Adjust invalid tags. E.g., 'O' followed by 'I-POS'.
        for i, tag in enumerate(mv_expert_tags):
            if i == 0 and 'I' in tag:
                mv_expert_tags[i] = tag.replace('I', 'B')
            elif 'O' == mv_expert_tags[i - 1] and 'I' in tag:
                mv_expert_tags[i] = tag.replace('I', 'B')
            elif 'B-POS' == mv_expert_tags[i - 1] and 'B-POS' == tag:
                mv_expert_tags[i] = tag.replace('B', 'I')
            elif 'B-NEG' == mv_expert_tags[i - 1] and 'B-NEG' == tag:
                mv_expert_tags[i] = tag.replace('B', 'I')

        return mv_expert_tags

    def get_mv_expert_tags_from_original_dataset(self, sentence_id):
        mv_expert_tags = []
        votes = [{} for _ in self.utils.sent2silver_tags[sentence_id]]
        for _, annotation in self.utils.sent2all_tags[sentence_id].items():
            for i, tag in enumerate(annotation):
                if tag not in votes[i].keys():
                    votes[i][tag] = 0
                votes[i][tag] += 1

        # Select tags according to votes.
        if self.utils.allow_fake_annotations:
            voter_num = self.utils.annotation_num_per_sentence
        else:
            voter_num = min(len(self.utils.sent2workers[sentence_id]), self.utils.annotation_num_per_sentence)
        self.sent2expert_tag_ratio[sentence_id] = 0
        for i, vote in enumerate(votes):
            sorted_vote = {k: v for k, v in sorted(vote.items(), key=lambda item: item[1], reverse=True)}
            if list(sorted_vote.items())[0][1] / voter_num >= self.acceptable_mv_percentage and voter_num != 1:
                mv_expert_tags.append(list(sorted_vote.items())[0][0])
            else:
                mv_expert_tags.append(self.utils.sent2silver_tags[sentence_id][i])
                self.sent2expert_tag_ratio[sentence_id] += 1
        if self.sent2expert_tag_ratio[sentence_id] > 0:
            self.expert_use_num += 1
        self.sent2expert_tag_ratio[sentence_id] /= len(self.utils.sent2silver_tags[sentence_id])

        # Adjust invalid tags. E.g., 'O' followed by 'I-POS'.
        for i, tag in enumerate(mv_expert_tags):
            if i == 0 and 'I' in tag:
                mv_expert_tags[i] = tag.replace('I', 'B')
            elif 'O' == mv_expert_tags[i - 1] and 'I' in tag:
                mv_expert_tags[i] = tag.replace('I', 'B')
            elif 'B-POS' == mv_expert_tags[i - 1] and 'B-POS' == tag:
                mv_expert_tags[i] = tag.replace('B', 'I')
            elif 'B-NEG' == mv_expert_tags[i - 1] and 'B-NEG' == tag:
                mv_expert_tags[i] = tag.replace('B', 'I')

        return mv_expert_tags


class KappaAggManager(MVExpertManager):

    def __init__(self, workers: list[Worker], utils: Utils, agg_method: str = 'mv', fleiss_kappa_threshold: float = 0.5,
                 **kwargs):
        super().__init__(workers, utils, **kwargs)

        self.fleiss_kappa_threshold = fleiss_kappa_threshold

        assert agg_method in ['mv', 'bsc']
        self.agg_method = agg_method

        self.sent2kappa: dict[int, float] = {}

    @property
    def run_name(self):
        return f'{self.__class__.__name__}' \
               f'-{self.utils.metrics_type}' \
               f'-aps={self.utils.annotation_num_per_sentence}' \
               f'-wn={self.selected_worker_num}' \
               f'-kt={self.fleiss_kappa_threshold}' \
               f'-us={self.ucb_scale}'

    def run_one_step(self) -> NoReturn:
        """
        Select the best workers according to their current average F-1 scores and let them annotate sentences.
        """
        self.t += 1

        is_eval_step = self.time_since_last_evaluation == self.evaluation_interval

        if is_eval_step:
            # Select the best workers with consideration of upper confidence bounds.
            worker2score: dict[Worker, float] = {
                worker: worker.average_f1 + self.ucb_scale * np.sqrt(3 * np.log(self.t) / (2 * worker.annotation_num))
                for worker in self.workers if worker not in self.exhausted_workers
            }

            # Record confidence bounds
            for worker in self.workers:
                if worker not in self.exhausted_workers:
                    self.confidence_bounds.append(
                        self.ucb_scale * np.sqrt(3 * np.log(self.t) / (2 * worker.annotation_num)))

            selected_workers: list[Worker] = [
                worker
                for worker, _ in
                sorted(worker2score.items(), key=lambda item: item[1], reverse=True)[:self.selected_worker_num]
            ]

            self.last_selection = selected_workers
            self.time_since_last_evaluation = 0
        else:
            # Use last selection.
            selected_workers = self.last_selection

            self.time_since_last_evaluation += 1

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            sent_anno_num += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=False,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)

                # Check if all annotations on this sentence are done.
                if self.utils.allow_fake_annotations:
                    voter_num = self.utils.annotation_num_per_sentence
                else:
                    voter_num = min(len(self.utils.sent2workers[sentence_id]), self.utils.annotation_num_per_sentence)
                if len(self.annotations[sentence_id]) == voter_num:
                    # Update f1 score records of workers.
                    if self.agg_method == 'mv':
                        answer_tags = self.get_mv_tags(sentence_id)
                    else:
                        answer_tags = self.get_bsc_tags(sentence_id)

                    fleiss_kappa = self.utils.metrics.fleiss_kappa(self.annotations[sentence_id])

                    self.sent2kappa[sentence_id] = fleiss_kappa

                    if fleiss_kappa < self.fleiss_kappa_threshold:
                        answer_tags = self.utils.sent2silver_tags[sentence_id]
                        self.expert_use_num += 1
                        self.sent2eval_type[sentence_id] = 1
                    else:
                        self.sent2eval_type[sentence_id] = 2

                    if is_eval_step:
                        for worker, annotation in zip(self.annotation_workers[sentence_id],
                                                      self.annotations[sentence_id]):
                            if self.utils.metrics_type == 'pearson':
                                worker.f1s.append(self.utils.metrics.pearson_corr_coef([answer_tags], [annotation]))
                            else:
                                worker.f1s.append(self.utils.metrics.f1_score([answer_tags], [annotation]))

                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)

    def get_bsc_tags(self, sentence_id):
        return []

    def get_mv_tags(self, sentence_id):
        mv_tags = []
        votes = [{} for _ in self.utils.sent2silver_tags[sentence_id]]
        for annotation in self.annotations[sentence_id]:
            for i, tag in enumerate(annotation):
                if tag not in votes[i].keys():
                    votes[i][tag] = 0
                votes[i][tag] += 1

        # Select tags according to votes.
        for i, vote in enumerate(votes):
            sorted_vote = {k: v for k, v in sorted(vote.items(), key=lambda item: item[1], reverse=True)}
            mv_tags.append(list(sorted_vote.items())[0][0])

        # Adjust invalid tags. E.g., 'O' followed by 'I-POS'.
        for i, tag in enumerate(mv_tags):
            if i == 0 and 'I' in tag:
                mv_tags[i] = tag.replace('I', 'B')
            elif 'O' == mv_tags[i - 1] and 'I' in tag:
                mv_tags[i] = tag.replace('I', 'B')
            elif 'B-POS' == mv_tags[i - 1] and 'B-POS' == tag:
                mv_tags[i] = tag.replace('B', 'I')
            elif 'B-NEG' == mv_tags[i - 1] and 'B-NEG' == tag:
                mv_tags[i] = tag.replace('B', 'I')

        return mv_tags


class MVManager(Manager):
    """
    Use majority voting only.
    """

    def __init__(self, workers: list[Worker], utils: Utils, **kwargs):
        super().__init__(workers, utils, **kwargs)
        self.annotation_workers: dict[int, list[Worker]] = {}

        self.sent2is_mv_tags_correct: dict[int, bool] = {}  # Sentence level correct
        self.sent2mv_tags_correct_ratio: dict[int, float] = {}  # Token level correct
        self.sent2mv_spans_correct_ratio: dict[int, float] = {}  # Span level correct

        self.confidence_bounds: list[float] = []

    def initialize(self) -> NoReturn:
        """
        Before normal rounds, let each worker annotate once for reference in later selections.
        """
        for worker in self.workers:
            self.t += 1
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            annotations, f1, sent2fake_annotation_num, _ = worker.annotate(sentence_ids, split_spans=self.split_spans)
            for sentence_id, annotation in zip(sentence_ids, annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
            self.selection_history.append(worker.id)

    def run_one_step(self) -> NoReturn:
        """
        Select the best workers according to their current average F-1 scores and let them annotate sentences.
        """
        self.t += 1

        # Select the best workers with consideration of upper confidence bounds.
        worker2score: dict[Worker, float] = {
            worker: worker.average_f1 + self.ucb_scale * np.sqrt(3 * np.log(self.t) / (2 * worker.annotation_num))
            for worker in self.workers if worker not in self.exhausted_workers
        }

        # Record confidence bounds
        for worker in self.workers:
            if worker not in self.exhausted_workers:
                self.confidence_bounds.append(
                    self.ucb_scale * np.sqrt(3 * np.log(self.t) / (2 * worker.annotation_num)))

        selected_workers: list[Worker] = [
            worker
            for worker, _ in
            sorted(worker2score.items(), key=lambda item: item[1], reverse=True)[:self.selected_worker_num]
        ]

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            sent_anno_num += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=False,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)

                # Check if all annotations on this sentence are done.
                if self.utils.allow_fake_annotations:
                    voter_num = self.utils.annotation_num_per_sentence
                else:
                    voter_num = min(len(self.utils.sent2workers[sentence_id]), self.utils.annotation_num_per_sentence)
                if len(self.annotations[sentence_id]) == voter_num:
                    # Update f1 score records of workers.
                    mv_tags = self.get_mv_tags(sentence_id)
                    for worker, annotation in zip(self.annotation_workers[sentence_id], self.annotations[sentence_id]):
                        if self.utils.metrics_type == 'pearson':
                            worker.f1s.append(self.utils.metrics.pearson_corr_coef([mv_tags], [annotation]))
                        else:
                            worker.f1s.append(self.utils.metrics.f1_score([mv_tags], [annotation]))

                    # Check if mv_tags is the same as the real expert tags.
                    self.sent2mv_tags_correct_ratio[sentence_id] = 0
                    is_current_span_correct = False
                    span_num = 0
                    correct_span_num = 0
                    for mv_tag, expert_tag in zip(mv_tags, self.utils.sent2silver_tags[sentence_id]):
                        if expert_tag.startswith('B'):
                            span_num += 1
                            if is_current_span_correct:
                                correct_span_num += 1
                            is_current_span_correct = True
                        if mv_tag == expert_tag:
                            self.sent2mv_tags_correct_ratio[sentence_id] += 1
                        else:
                            is_current_span_correct = False
                    self.sent2mv_spans_correct_ratio[sentence_id] = correct_span_num / span_num
                    self.sent2is_mv_tags_correct[sentence_id] = self.sent2mv_tags_correct_ratio[sentence_id] == len(
                        mv_tags)
                    self.sent2mv_tags_correct_ratio[sentence_id] /= len(mv_tags)

                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)

    def get_mv_tags(self, sentence_id):
        mv_tags = []
        votes = [{} for _ in self.utils.sent2silver_tags[sentence_id]]
        for annotation in self.annotations[sentence_id]:
            for i, tag in enumerate(annotation):
                if tag not in votes[i].keys():
                    votes[i][tag] = 0
                votes[i][tag] += 1

        # Select tags according to votes.
        for i, vote in enumerate(votes):
            sorted_vote = {k: v for k, v in sorted(vote.items(), key=lambda item: item[1], reverse=True)}
            mv_tags.append(list(sorted_vote.items())[0][0])

        # Adjust invalid tags. E.g., 'O' followed by 'I-POS'.
        for i, tag in enumerate(mv_tags):
            if i == 0 and 'I' in tag:
                mv_tags[i] = tag.replace('I', 'B')
            elif 'O' == mv_tags[i - 1] and 'I' in tag:
                mv_tags[i] = tag.replace('I', 'B')
            elif 'B-POS' == mv_tags[i - 1] and 'B-POS' == tag:
                mv_tags[i] = tag.replace('B', 'I')
            elif 'B-NEG' == mv_tags[i - 1] and 'B-NEG' == tag:
                mv_tags[i] = tag.replace('B', 'I')

        return mv_tags

    def get_mv_tags_from_original_dataset(self, sentence_id):
        mv_tags = []
        votes = [{} for _ in self.utils.sent2silver_tags[sentence_id]]
        for _, annotation in self.utils.sent2all_tags[sentence_id].items():
            for i, tag in enumerate(annotation):
                if tag not in votes[i].keys():
                    votes[i][tag] = 0
                votes[i][tag] += 1

        # Select tags according to votes.
        for i, vote in enumerate(votes):
            sorted_vote = {k: v for k, v in sorted(vote.items(), key=lambda item: item[1], reverse=True)}
            mv_tags.append(list(sorted_vote.items())[0][0])

        # Adjust invalid tags. E.g., 'O' followed by 'I-POS'.
        for i, tag in enumerate(mv_tags):
            if i == 0 and 'I' in tag:
                mv_tags[i] = tag.replace('I', 'B')
            elif 'O' == mv_tags[i - 1] and 'I' in tag:
                mv_tags[i] = tag.replace('I', 'B')
            elif 'B-POS' == mv_tags[i - 1] and 'B-POS' == tag:
                mv_tags[i] = tag.replace('B', 'I')
            elif 'B-NEG' == mv_tags[i - 1] and 'B-NEG' == tag:
                mv_tags[i] = tag.replace('B', 'I')

        return mv_tags


class EpsilonGreedyManager(KappaAggManager):

    def __init__(self, workers: list[Worker], utils: Utils, epsilon: float = 0.5, **kwargs):
        super().__init__(workers, utils, **kwargs)
        self.epsilon = epsilon

    @property
    def run_name(self):
        return f'{self.__class__.__name__}' \
               f'-{self.utils.metrics_type}' \
               f'-aps={self.utils.annotation_num_per_sentence}' \
               f'-wn={self.selected_worker_num}' \
               f'-kt={self.fleiss_kappa_threshold}' \
               f'-ep={self.epsilon}'

    def run_one_step(self) -> NoReturn:
        """
        Select the best workers according to their current average F-1 scores and let them annotate sentences.
        """
        self.t += 1

        is_eval_step = self.time_since_last_evaluation == self.evaluation_interval

        if is_eval_step:
            if random.random() < self.epsilon:
                # Randomly select workers
                random.shuffle(self.workers)
                selected_workers = self.workers[:self.selected_worker_num]
            else:
                # Select the observed best workers.
                worker2score: dict[Worker, float] = {
                    worker: worker.average_f1
                    for worker in self.workers if worker not in self.exhausted_workers
                }

                selected_workers: list[Worker] = [
                    worker
                    for worker, _ in
                    sorted(worker2score.items(), key=lambda item: item[1], reverse=True)[:self.selected_worker_num]
                ]

            self.last_selection = selected_workers
            self.time_since_last_evaluation = 0
        else:
            # Use last selection.
            selected_workers = self.last_selection

            self.time_since_last_evaluation += 1

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            sent_anno_num += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=False,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)

                # Check if all annotations on this sentence are done.
                if self.utils.allow_fake_annotations:
                    voter_num = self.utils.annotation_num_per_sentence
                else:
                    voter_num = min(len(self.utils.sent2workers[sentence_id]), self.utils.annotation_num_per_sentence)
                if len(self.annotations[sentence_id]) == voter_num:
                    # Update f1 score records of workers.
                    if self.agg_method == 'mv':
                        answer_tags = self.get_mv_tags(sentence_id)
                    else:
                        answer_tags = self.get_bsc_tags(sentence_id)

                    fleiss_kappa = self.utils.metrics.fleiss_kappa(self.annotations[sentence_id])

                    self.sent2kappa[sentence_id] = fleiss_kappa

                    if fleiss_kappa < self.fleiss_kappa_threshold:
                        answer_tags = self.utils.sent2silver_tags[sentence_id]
                        self.expert_use_num += 1
                        self.sent2eval_type[sentence_id] = 1
                    else:
                        self.sent2eval_type[sentence_id] = 2

                    if is_eval_step:
                        for worker, annotation in zip(self.annotation_workers[sentence_id],
                                                      self.annotations[sentence_id]):
                            if self.utils.metrics_type == 'pearson':
                                worker.f1s.append(self.utils.metrics.pearson_corr_coef([answer_tags], [annotation]))
                            else:
                                worker.f1s.append(self.utils.metrics.f1_score([answer_tags], [annotation]))

                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)


class ThompsonSamplingManager(KappaAggManager):

    def __init__(self, workers: list[Worker], utils: Utils, sample_num: int = 100, **kwargs):
        super().__init__(workers, utils, **kwargs)
        self.sample_num = sample_num
        self.worker2alpha: dict[Worker, float] = {worker: 1.0 for worker in self.workers}
        self.worker2beta: dict[Worker, float] = {worker: 1.0 for worker in self.workers}

    @property
    def run_name(self):
        return f'{self.__class__.__name__}' \
               f'-{self.utils.metrics_type}' \
               f'-aps={self.utils.annotation_num_per_sentence}' \
               f'-wn={self.selected_worker_num}' \
               f'-kt={self.fleiss_kappa_threshold}' \
               f'-sn={self.sample_num}'

    def run_one_step(self) -> NoReturn:
        """
        Select the best workers according to their current average F-1 scores and let them annotate sentences.
        """
        self.t += 1

        is_eval_step = self.time_since_last_evaluation == self.evaluation_interval

        if is_eval_step:
            worker2sample_reward: dict[Worker, float] = {}
            for worker in self.worker2alpha.keys():
                worker2sample_reward[worker] = float(np.sum(np.random.beta(a=self.worker2alpha[worker],
                                                                           b=self.worker2beta[worker],
                                                                           size=self.sample_num)))
            worker2sample_reward = dict(sorted(worker2sample_reward.items(), key=lambda item: item[1], reverse=True))

            selected_workers = list(worker2sample_reward.keys())[:self.selected_worker_num]

            self.last_selection = selected_workers
            self.time_since_last_evaluation = 0
        else:
            # Use last selection.
            selected_workers = self.last_selection

            self.time_since_last_evaluation += 1

        # If all workers are exhausted, end worker-selection.
        if not selected_workers:
            return

        # Let the workers annotate new sentences.
        step_f1 = 0
        actual_worker_num = 0
        sent_anno_num = 0
        last_sentence_ids = []
        for worker in selected_workers:
            if self.utils.annotation_num_per_sentence >= sent_anno_num:
                sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
                last_sentence_ids = list(sentence_ids)
                sent_anno_num = 0
            else:
                sentence_ids = list(last_sentence_ids)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            sent_anno_num += 1
            annotations, f1, sent2fake_annotation_num, is_fake_annotations = worker.annotate(sentence_ids,
                                                                                             update_f1=False,
                                                                                             split_spans=self.split_spans)
            for sentence_id, annotation, is_fake_annotation in zip(sentence_ids, annotations, is_fake_annotations):
                self.annotations[sentence_id].append(annotation)
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)
                self.sent2fake_annotation_num[sentence_id] += sent2fake_annotation_num[sentence_id]
                if is_fake_annotation:
                    if sentence_id not in self.fake_annotations.keys():
                        self.fake_annotations[sentence_id] = []
                    self.fake_annotations[sentence_id].append(annotation)

                # Check if all annotations on this sentence are done.
                if self.utils.allow_fake_annotations:
                    voter_num = self.utils.annotation_num_per_sentence
                else:
                    voter_num = min(len(self.utils.sent2workers[sentence_id]), self.utils.annotation_num_per_sentence)
                if len(self.annotations[sentence_id]) == voter_num:
                    # Update f1 score records of workers.
                    if self.agg_method == 'mv':
                        answer_tags = self.get_mv_tags(sentence_id)
                    else:
                        answer_tags = self.get_bsc_tags(sentence_id)

                    fleiss_kappa = self.utils.metrics.fleiss_kappa(self.annotations[sentence_id])

                    self.sent2kappa[sentence_id] = fleiss_kappa

                    if fleiss_kappa < self.fleiss_kappa_threshold:
                        answer_tags = self.utils.sent2silver_tags[sentence_id]
                        self.expert_use_num += 1
                        self.sent2eval_type[sentence_id] = 1
                    else:
                        self.sent2eval_type[sentence_id] = 2

                    if is_eval_step:
                        for worker, annotation in zip(self.annotation_workers[sentence_id],
                                                      self.annotations[sentence_id]):
                            if self.utils.metrics_type == 'pearson':
                                score = self.utils.metrics.pearson_corr_coef([answer_tags], [annotation])
                                worker.f1s.append(score)
                            else:
                                score = self.utils.metrics.f1_score([answer_tags], [annotation])
                                worker.f1s.append(score)

                            self.worker2alpha[worker] += score
                            self.worker2beta[worker] += 1 - score

                step_f1 += f1

        self.selection_history.append(selected_workers)
        # if actual_worker_num < self.selected_worker_num:
        #     return
        if actual_worker_num == 0:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1 * self.selected_worker_num / actual_worker_num)

def log(content: str = '\n', filepath: str = 'out/stdout.txt'):
    print(content)
    with open(filepath, 'a') as out:
        out.write(f'{content}\n')


def run_average(average_over: int = 100,
                num_steps: int = 10000,
                annotation_num_per_sentence: int = 1,
                allow_exhaustion: bool = True,
                evaluation_interval: int = 10,
                manager_type: str = 'normal',
                allow_fake_annotations: bool = False,
                only_fake_annotations: bool = False,
                metrics_type: str = 'span_exact',
                split_spans: bool = False,
                ucb_scale: float = 1.0,
                acceptable_mv_char_error_nums: int = 0,
                acceptable_mv_percentage: float = 0.67,
                fleiss_kappa_threshold: float = 0.5,
                agg_method: str = 'mv',
                use_fake_annotation_cache: bool = False,
                epsilon: float = 0.5,
                selected_worker_num: int = 20,
                use_gold_expert: bool = False) -> NoReturn:
    utils = None
    if manager_type == 'normal':
        manager_class = Manager
    elif manager_type == 'mv_expert':
        manager_class = MVExpertManager
    elif manager_type == 'mv':
        manager_class = MVManager
    elif manager_type == 'kappa_agg':
        manager_class = KappaAggManager
    elif manager_type == 'best':
        manager_class = BestManager
    elif manager_type == 'worst':
        manager_class = WorstManager
    elif manager_type == 'random':
        manager_class = RandomManager
    elif manager_type == 'epsilon_greedy':
        manager_class = EpsilonGreedyManager
    elif manager_type == 'thompson_sampling':
        manager_class = ThompsonSamplingManager
    else:
        raise ValueError(f'Unknown manager type {manager_type} is given.')

    span_exact_precisions = []
    span_exact_recalls = []
    span_exact_f1s = []

    span_proportional_precisions = []
    span_proportional_recalls = []
    span_proportional_f1s = []

    token_precisions = []
    token_recalls = []
    token_f1s = []

    pccs = []

    fake_span_exact_precisions = []
    fake_span_exact_recalls = []
    fake_span_exact_f1s = []

    fake_span_proportional_precisions = []
    fake_span_proportional_recalls = []
    fake_span_proportional_f1s = []

    fake_token_precisions = []
    fake_token_recalls = []
    fake_token_f1s = []

    fake_pccs = []

    expert_use_nums = []
    expert_tag_ratios = []

    mv_correct_sentence_level = []
    mv_correct_token_level = []
    mv_correct_span_level = []

    fake_precision_errors = []
    fake_recall_errors = []
    fake_f1_errors = []
    fake_pcc_errors = []
    error2times = {}

    confidence_bounds = []

    fake_annotation_num_per_sentence = []

    regret_histories = []

    round_sent2annotations = []
    round_sent2workers = []
    round_sent2eval_type = []

    memory = None

    annotation_num = 0
    for average_round in trange(average_over, desc=f'Averaging over {average_over} trials: ', position=0, ncols=80):
        utils = Utils(annotation_num_per_sentence,
                      allow_fake_annotations=allow_fake_annotations,
                      only_fake_annotations=only_fake_annotations,
                      metrics_type=metrics_type,
                      use_fake_annotation_cache=use_fake_annotation_cache,
                      use_gold_expert=use_gold_expert,
                      memory=memory)
        memory = {
            'data': utils.data,
            'sent2silver_tags': utils.sent2silver_tags
        }
        workers = [Worker(worker_id, utils) for worker_id in utils.worker_ids]
        manager = manager_class(workers, utils,
                                allow_exhaustion=allow_exhaustion,
                                evaluation_interval=evaluation_interval,
                                split_spans=split_spans,
                                ucb_scale=ucb_scale,
                                selected_worker_num=selected_worker_num)
        if issubclass(manager_class, MVExpertManager):
            manager.acceptable_mv_char_error_nums = acceptable_mv_char_error_nums
            manager.acceptable_mv_percentage = acceptable_mv_percentage
        if issubclass(manager_class, KappaAggManager):
            manager.fleiss_kappa_threshold = fleiss_kappa_threshold
            manager.agg_method = agg_method
        if issubclass(manager_class, EpsilonGreedyManager):
            manager.epsilon = epsilon
        manager.run(num_steps=num_steps)

        token_precision, token_recall, token_f1, annotation_num, \
        fake_token_precision, fake_token_recall, fake_token_f1, \
        pcc, fake_pcc = manager.evaluate(
            dump_results=False,
            metrics_type='token')

        span_exact_precision, span_exact_recall, span_exact_f1, annotation_num, \
        fake_span_exact_precision, fake_span_exact_recall, fake_span_exact_f1, \
        pcc, fake_pcc = manager.evaluate(
            dump_results=False,
            metrics_type='span_exact')

        span_proportional_precision, span_proportional_recall, span_proportional_f1, annotation_num, \
        fake_span_proportional_precision, fake_span_proportional_recall, fake_span_proportional_f1, \
        pcc, fake_pcc = manager.evaluate(
            dump_results=False,
            metrics_type='span_proportional')

        token_precisions.append(token_precision * 100)
        token_recalls.append(token_recall * 100)
        token_f1s.append(token_f1 * 100)

        span_exact_precisions.append(span_exact_precision * 100)
        span_exact_recalls.append(span_exact_recall * 100)
        span_exact_f1s.append(span_exact_f1 * 100)

        span_proportional_precisions.append(span_proportional_precision * 100)
        span_proportional_recalls.append(span_proportional_recall * 100)
        span_proportional_f1s.append(span_proportional_f1 * 100)

        pccs.append(pcc)

        fake_token_precisions.append(fake_token_precision * 100)
        fake_token_recalls.append(fake_token_recall * 100)
        fake_token_f1s.append(fake_token_f1 * 100)

        fake_span_exact_precisions.append(fake_span_exact_precision * 100)
        fake_span_exact_recalls.append(fake_span_exact_recall * 100)
        fake_span_exact_f1s.append(fake_span_exact_f1 * 100)

        fake_span_proportional_precisions.append(fake_span_proportional_precision * 100)
        fake_span_proportional_recalls.append(fake_span_proportional_recall * 100)
        fake_span_proportional_f1s.append(fake_span_proportional_f1 * 100)

        fake_pccs.append(fake_pcc)

        fake_precision_errors.extend(utils.fake_precision_errors)
        fake_recall_errors.extend(utils.fake_recall_errors)
        fake_f1_errors.extend(utils.fake_f1_errors)
        fake_pcc_errors.extend(utils.fake_pcc_errors)

        # round_sent2annotations.append(manager.annotations)
        # round_sent2workers.append(manager.annotation_workers)
        # round_sent2eval_type.append(manager.sent2eval_type)

        for error, times in utils.error2times.items():
            if error not in error2times.keys():
                error2times[error] = 0
            error2times[error] += times

        if manager_class == MVExpertManager:
            expert_use_nums.append(manager.expert_use_num)
            expert_tag_ratios.extend(manager.sent2expert_tag_ratio.values())

            mv_correct_sentence_level.append(
                sum(map(lambda is_correct: 1 if is_correct else 0, list(manager.sent2is_mv_tags_correct.values())))
                / len(utils.sent2workers)
            )
            mv_correct_token_level.append(sum(manager.sent2mv_tags_correct_ratio.values()) / len(utils.sent2workers))

            confidence_bounds.extend(manager.confidence_bounds)
        elif manager_class == MVManager:
            mv_correct_sentence_level.append(
                sum(map(lambda is_correct: 1 if is_correct else 0, list(manager.sent2is_mv_tags_correct.values())))
                / len(utils.sent2workers)
            )
            mv_correct_token_level.append(sum(manager.sent2mv_tags_correct_ratio.values()) / len(utils.sent2workers))
            mv_correct_span_level.append(sum(manager.sent2mv_spans_correct_ratio.values()) / len(utils.sent2workers))

            confidence_bounds.extend(manager.confidence_bounds)
        elif manager_class == KappaAggManager or manager_class == EpsilonGreedyManager:
            expert_use_nums.append(manager.expert_use_num)

        fake_annotation_num_per_sentence.append(np.mean(list(manager.sent2fake_annotation_num.values())))

        manager.dump_annotations()

        regret_histories.append(manager.regret_history)

    logf = partial(log, filepath=f'out/std_logs/oei/{manager.run_name}.txt')
    with open(f'out/std_logs/oei/{manager.run_name}.txt', 'w') as _:
        pass

    logf(f'Evaluation over {average_over} trials.')
    logf(f'Produced {annotation_num} annotations in total.\n')

    table = PrettyTable()
    table.field_names = ['Metric', 'Token-level', 'Span-level Exact', 'Span-level Proportional']
    table.add_row([
        'Precision',
        f'{np.mean(token_precisions):.02f} / {np.var(token_precisions):.02f}',
        f'{np.mean(span_exact_precisions):.02f} / {np.var(span_exact_precisions):.02f}',
        f'{np.mean(span_proportional_precisions):.02f} / {np.var(span_proportional_precisions):.02f}',
    ])
    table.add_row([
        'Recall',
        f'{np.mean(token_recalls):.02f} / {np.var(token_recalls):.02f}',
        f'{np.mean(span_exact_recalls):.02f} / {np.var(span_exact_recalls):.02f}',
        f'{np.mean(span_proportional_recalls):.02f} / {np.var(span_proportional_recalls):.02f}',
    ])
    table.add_row([
        'F-1',
        f'{np.mean(token_f1s):.02f} / {np.var(token_f1s):.02f}',
        f'{np.mean(span_exact_f1s):.02f} / {np.var(span_exact_f1s):.02f}',
        f'{np.mean(span_proportional_f1s):.02f} / {np.var(span_proportional_f1s):.02f}',
    ])
    table.align['Metric'] = 'l'
    table.align['Token-level'] = 'r'
    table.align['Span-level Exact'] = 'r'
    table.align['Span-level Proportional'] = 'r'

    logf('Actual Scores')
    logf(table)
    with open(f'out/f1_logs/oei/{manager.run_name}.txt', 'w') as out:
        out.write(f'{np.mean(span_proportional_f1s):.02f} / {np.var(span_proportional_f1s):.02f}')

    table = PrettyTable()
    table.field_names = ['Metric', 'Token-level', 'Span-level Exact', 'Span-level Proportional']
    table.add_row([
        'Precision',
        f'{np.mean(fake_token_precisions):.02f} / {np.var(fake_token_precisions):.02f}',
        f'{np.mean(fake_span_exact_precisions):.02f} / {np.var(fake_span_exact_precisions):.02f}',
        f'{np.mean(fake_span_proportional_precisions):.02f} / {np.var(fake_span_proportional_precisions):.02f}',
    ])
    table.add_row([
        'Recall',
        f'{np.mean(fake_token_recalls):.02f} / {np.var(fake_token_recalls):.02f}',
        f'{np.mean(fake_span_exact_recalls):.02f} / {np.var(fake_span_exact_recalls):.02f}',
        f'{np.mean(fake_span_proportional_recalls):.02f} / {np.var(fake_span_proportional_recalls):.02f}',
    ])
    table.add_row([
        'F-1',
        f'{np.mean(fake_token_f1s):.02f} / {np.var(fake_token_f1s):.02f}',
        f'{np.mean(fake_span_exact_f1s):.02f} / {np.var(fake_span_exact_f1s):.02f}',
        f'{np.mean(fake_span_proportional_f1s):.02f} / {np.var(fake_span_proportional_f1s):.02f}',
    ])
    table.align['Metric'] = 'l'
    table.align['Token-level'] = 'r'
    table.align['Span-level Exact'] = 'r'
    table.align['Span-level Proportional'] = 'r'

    logf('Fake Scores')
    logf(table)

    if manager_class == MVExpertManager:
        logf(f'Experts used in {np.mean(expert_use_nums) + len(utils.worker_ids):.02f} / {len(utils.sent2workers)}'
             f' = {np.mean(expert_use_nums) / len(utils.sent2workers) * 100:.02f} sentences.')
        logf(
            f'Expert tag ratio in sentence: {np.mean(expert_tag_ratios) * 100:.02f} / {np.var(expert_tag_ratios):.02f}\n')

        logf(f'Sentence-level mv correct ratio: {np.mean(mv_correct_sentence_level) * 100:.02f}')
        logf(f'Token-level mv correct ratio: {np.mean(mv_correct_token_level) * 100:.02f}\n')
        with open(f'out/expct_logs/oei/{manager.run_name}.txt', 'w') as out:
            out.write(f'{np.mean(expert_use_nums) / len(utils.sent2workers) * 100:.02f}')

        # logf(f'Confidence bound stats:')
        # logf(f'Min: {min(confidence_bounds)}')
        # logf(f'Max: {max(confidence_bounds)}')
        # logf(f'Avg: {np.mean(confidence_bounds)}')
        # logf()
    elif manager_class == MVManager:
        logf(f'Sentence-level mv correct ratio: {np.mean(mv_correct_sentence_level) * 100:.02f}')
        logf(f'Token-level mv correct ratio: {np.mean(mv_correct_token_level) * 100:.02f}')
        logf(f'Span-level mv correct ratio: {np.mean(mv_correct_span_level) * 100:.02f}\n')

        # logf(f'Confidence bound stats:')
        # logf(f'Min: {min(confidence_bounds)}')
        # logf(f'Max: {max(confidence_bounds)}')
        # logf(f'Avg: {np.mean(confidence_bounds)}')
        # logf()
    elif manager_class == KappaAggManager or manager_class == EpsilonGreedyManager:
        logf(f'Experts used in {np.mean(expert_use_nums) + len(utils.worker_ids):.02f} / {len(utils.sent2workers)}'
             f' = {np.mean(expert_use_nums) / len(utils.sent2workers) * 100:.02f} sentences.')
        with open(f'out/expct_logs/oei/{manager.run_name}.txt', 'w') as out:
            out.write(f'{np.mean(expert_use_nums) / len(utils.sent2workers) * 100:.02f}')

        # min_greater_idx = -1
        # for i, f1 in enumerate(span_proportional_f1s):
        #     if f1 < np.mean(span_proportional_f1s):
        #         continue
        #     if min_greater_idx == -1:
        #         min_greater_idx = i
        #         continue
        #     if f1 < span_proportional_f1s[min_greater_idx]:
        #         min_greater_idx = i
        # manager.dump_annotations(annotations=round_sent2annotations[min_greater_idx],
        #                          annotation_workers=round_sent2workers[min_greater_idx],
        #                          annotation_eval_types=round_sent2eval_type[min_greater_idx],
        #                          filename_suffix=f'-f{span_proportional_f1s[min_greater_idx]:.02f}'
        #                                          f'-m{np.mean(span_proportional_f1s):.02f}')

    logf(f'Fake Annotation Error')
    logf(f'P:  {np.mean(fake_precision_errors):.03f} / {np.var(fake_precision_errors):.03f}')
    logf(f'R:  {np.mean(fake_recall_errors):.03f} / {np.var(fake_recall_errors):.03f}')
    logf(f'F1: {np.mean(fake_f1_errors):.03f} / {np.var(fake_f1_errors):.03f}')
    if utils.metrics.metrics_type == 'pearson':
        logf(f'PCC: {np.mean(fake_pcc_errors):.03f} / {np.var(fake_pcc_errors):.03f}')

    # logf('Error: Times')
    # for error, times in sorted(utils.error2times.items(), key=lambda item: item[0]):
    #     logf(f'{error}: {times / average_over:.02f}')

    logf(f'Fake annotation num per sentence: {np.mean(fake_annotation_num_per_sentence):.02f}')

    min_length = len(regret_histories[0])
    for history in regret_histories:
        if min_length > len(history):
            min_length = len(history)
    for i, history in enumerate(regret_histories):
        regret_histories[i] = history[:min_length]

    min_len = min([len(history) for history in regret_histories])
    regret_histories = [history[:min_len] for history in regret_histories]
    manager.plot_regret(regret_history=np.mean(regret_histories, axis=0))


def run_once(num_steps: int = 10000,
             annotation_num_per_sentence: int = 1,
             allow_exhaustion: bool = True,
             evaluation_interval=10,
             manager_type: str = 'normal',
             allow_fake_annotations: bool = False,
             only_fake_annotations: bool = False,
             metrics_type: str = 'span_exact',
             split_spans: bool = False,
             ucb_scale: float = 1.0,
             acceptable_mv_char_error_nums: int = 0,
             acceptable_mv_percentage: float = 0.67,
             fleiss_kappa_threshold: float = 0.5,
             agg_method: str = 'mv',
             use_fake_annotation_cache: bool = False,
             epsilon: float = 0.5,
             use_gold_expert: bool = False) -> NoReturn:
    global utils
    if manager_type == 'normal':
        manager_class = Manager
    elif manager_type == 'mv_expert':
        manager_class = MVExpertManager
    elif manager_type == 'mv':
        manager_class = MVManager
    elif manager_type == 'kappa_agg':
        manager_class = KappaAggManager
    elif manager_type == 'best':
        manager_class = BestManager
    elif manager_type == 'worst':
        manager_class = WorstManager
    elif manager_type == 'random':
        manager_class = RandomManager
    elif manager_type == 'epsilon_greedy':
        manager_class = EpsilonGreedyManager
    elif manager_type == 'thompson_sampling':
        manager_class = ThompsonSamplingManager
    else:
        raise ValueError(f'Unknown manager type {manager_type} is given.')
    utils = Utils(annotation_num_per_sentence=annotation_num_per_sentence,
                  allow_fake_annotations=allow_fake_annotations,
                  only_fake_annotations=only_fake_annotations,
                  metrics_type=metrics_type,
                  use_fake_annotation_cache=use_fake_annotation_cache,
                  use_gold_expert=use_gold_expert)
    workers = [Worker(worker_id, utils) for worker_id in utils.worker_ids]
    manager = manager_class(workers=workers,
                            utils=utils,
                            allow_exhaustion=allow_exhaustion,
                            evaluation_interval=evaluation_interval,
                            split_spans=split_spans,
                            ucb_scale=ucb_scale)
    if issubclass(manager_class, MVExpertManager):
        manager.acceptable_mv_char_error_nums = acceptable_mv_char_error_nums
        manager.acceptable_mv_percentage = acceptable_mv_percentage
    if issubclass(manager_class, KappaAggManager):
        manager.fleiss_kappa_threshold = fleiss_kappa_threshold
        manager.agg_method = agg_method
    if issubclass(manager_class, EpsilonGreedyManager):
        manager.epsilon = epsilon
    manager.run(num_steps=num_steps)

    token_precision, token_recall, token_f1, annotation_num, \
    fake_token_precision, fake_token_recall, fake_token_f1, pcc, fake_pcc = manager.evaluate(
        dump_results=False,
        metrics_type='token')
    span_exact_precision, span_exact_recall, span_exact_f1, annotation_num, \
    fake_span_exact_precision, fake_span_exact_recall, fake_span_exact_f1, pcc, fake_pcc = manager.evaluate(
        dump_results=False,
        metrics_type='span_exact')
    span_proportional_precision, span_proportional_recall, span_proportional_f1, annotation_num, \
    fake_span_proportional_precision, fake_span_proportional_recall, fake_span_proportional_f1, pcc, fake_pcc = manager.evaluate(
        metrics_type='span_proportional')

    manager.plot_regret()

    logf = partial(log, filepath=f'out/std_logs/oei/{manager.run_name}.txt')
    with open(f'out/std_logs/oei/{manager.run_name}.txt', 'w') as _:
        pass

    # manager.plot_sent_len2f1_dist()
    # utils.export_selections(manager.workers)
    logf(f'\nProduced {annotation_num} annotations in total.')
    logf(f'{annotation_num / len(utils.sent2workers):.02f} annotations per sentence on average.')

    table = PrettyTable()
    table.field_names = ['Metric', 'Token-level', 'Span-level Exact', 'Span-level Proportional']
    table.add_row(['Precision', f'{token_precision * 100:.02f}', f'{span_exact_precision * 100:.02f}',
                   f'{span_proportional_precision * 100:.02f}'])
    table.add_row(['Recall', f'{token_recall * 100:.02f}', f'{span_exact_recall * 100:.02f}',
                   f'{span_proportional_recall * 100:.02f}'])
    table.add_row(
        ['F-1', f'{token_f1 * 100:.02f}', f'{span_exact_f1 * 100:.02f}', f'{span_proportional_f1 * 100:.02f}'])
    table.align['Metric'] = 'l'
    table.align['Token-level'] = 'r'
    table.align['Span-level Exact'] = 'r'
    table.align['Span-level Proportional'] = 'r'
    logf('Actual Scores')
    logf(table)
    with open(f'out/f1_logs/oei/{manager.run_name}.txt', 'w') as out:
        out.write(f'{span_proportional_f1 * 100:.02f}')

    table = PrettyTable()
    table.field_names = ['Metric', 'Token-level', 'Span-level Exact', 'Span-level Proportional']
    table.add_row(['Precision', f'{fake_token_precision * 100:.02f}', f'{fake_span_exact_precision * 100:.02f}',
                   f'{fake_span_proportional_precision * 100:.02f}'])
    table.add_row(['Recall', f'{fake_token_recall * 100:.02f}', f'{fake_span_exact_recall * 100:.02f}',
                   f'{fake_span_proportional_recall * 100:.02f}'])
    table.add_row(
        ['F-1', f'{fake_token_f1 * 100:.02f}', f'{fake_span_exact_f1 * 100:.02f}',
         f'{fake_span_proportional_f1 * 100:.02f}'])
    table.align['Metric'] = 'l'
    table.align['Token-level'] = 'r'
    table.align['Span-level Exact'] = 'r'
    table.align['Span-level Proportional'] = 'r'
    logf('Fake Scores')
    logf(table)

    logf(f'Fake Annotation Error')
    logf(f'P:  {np.mean(utils.fake_precision_errors):.03f} / {np.var(utils.fake_precision_errors):.03f}')
    logf(f'R:  {np.mean(utils.fake_recall_errors):.03f} / {np.var(utils.fake_recall_errors):.03f}')
    logf(f'F1: {np.mean(utils.fake_f1_errors):.03f} / {np.var(utils.fake_f1_errors):.03f}')

    # logf('Error: Times')
    # for error, times in sorted(utils.error2times.items(), key=lambda item: item[0]):
    #     logf(f'{error}: {times}')
    # logf()

    if hasattr(manager, 'expert_use_num'):
        logf(f'Experts used in {manager.expert_use_num + len(utils.worker_ids):.02f} / {len(utils.sent2workers)}'
             f' = {manager.expert_use_num / len(utils.sent2workers) * 100:.02f} sentences.')
        with open(f'out/expct_logs/oei/{manager.run_name}.txt', 'w') as out:
            out.write(f'{manager.expert_use_num / len(utils.sent2workers) * 100:.02f}')

    manager.dump_annotations()

    # if hasattr(manager, 'sent2kappa'):
    #     with open('out/sent2kappa.txt', 'w') as writer:
    #         for sent, kappa in manager.sent2kappa.items():
    #             writer.write(f'{sent}, {kappa}\n')
    # logf(f'Confidence bound stats:')
    # logf(f'Min: {min(manager.confidence_bounds)}')
    # logf(f'Max: {max(manager.confidence_bounds)}')
    # logf(f'Avg: {np.mean(manager.confidence_bounds)}')


if __name__ == '__main__':
    run_once(num_steps=10000,
             annotation_num_per_sentence=4,
             allow_exhaustion=True,
             evaluation_interval=0,
             allow_fake_annotations=True,
             only_fake_annotations=False,
             manager_type='thompson_sampling',
             fleiss_kappa_threshold=0.4,
             agg_method='mv',
             use_gold_expert=True,
             metrics_type='span_proportional',
             use_fake_annotation_cache=True)

    # run_average(average_over=3,
    #             num_steps=10000,
    #             annotation_num_per_sentence=4,
    #             allow_exhaustion=True,
    #             evaluation_interval=0,
    #             allow_fake_annotations=True,
    #             only_fake_annotations=False,
    #             manager_type='kappa_agg',
    #             fleiss_kappa_threshold=0.4,
    #             agg_method='mv',
    #             use_gold_expert=True,
    #             metrics_type='span_proportional',
    #             use_fake_annotation_cache=True)

    # run_average(average_over=2,
    #             num_steps=10000,
    #             annotation_num_per_sentence=4,
    #             allow_exhaustion=True,
    #             evaluation_interval=0,
    #             allow_fake_annotations=True,
    #             only_fake_annotations=False,
    #             manager_type='epsilon_greedy',
    #             fleiss_kappa_threshold=0.4,
    #             agg_method='mv',
    #             epsilon=0.3,
    #             use_gold_expert=True,
    #             metrics_type='span_proportional',
    #             use_fake_annotation_cache=True)

    # run_once(num_steps=10000,
    #          annotation_num_per_sentence=10,
    #          allow_exhaustion=True,
    #          evaluation_interval=0,
    #          allow_fake_annotations=True,
    #          only_fake_annotations=False,
    #          manager_type='random',
    #          metrics_type='span_proportional',
    #          use_gold_expert=True,
    #          use_fake_annotation_cache=True)

    # run_average(average_over=1,
    #             num_steps=5000,
    #             annotation_num_per_sentence=2,
    #             allow_exhaustion=True,
    #             evaluation_interval=0,
    #             allow_fake_annotations=True,
    #             manager_type='mv_expert',
    #             metrics_type='span_proportional',
    #             split_spans=True)

    # 8047, 1.0 = 469 rounds
    # 15046, 1.87 = 819 rounds
    # 20805, 2.59 = 1107 rounds

    # num_steps = 5000
    # annotation_num_per_sentence=4
    # allow_exhaustion=True
    # evaluation_interval=0
    # allow_fake_annotations=True
    # manager_type='mv_expert'
    # metrics_type='span_proportional'
    # split_spans = False
    # ucb_scale = 1.0
    # #
    # utils = Utils(annotation_num_per_sentence=annotation_num_per_sentence,
    #               allow_fake_annotations=allow_fake_annotations,
    #               metrics_type=metrics_type)
    # workers = [Worker(worker_id, utils) for worker_id in utils.worker_ids]
    # manager = MVExpertManager(workers=workers,
    #                     utils=utils,
    #                     allow_exhaustion=allow_exhaustion,
    #                     evaluation_interval=evaluation_interval,
    #                     split_spans=split_spans,
    #                     ucb_scale=ucb_scale)
    #
    # for sentence_id in manager.utils.sent2all_tags.keys():
    #     manager.get_mv_expert_tags_from_original_dataset(sentence_id)
    #
    # logf(manager.expert_use_num)
