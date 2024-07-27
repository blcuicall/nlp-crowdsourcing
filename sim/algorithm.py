import random
from collections import OrderedDict
from functools import reduce
from typing import Union, NoReturn, Tuple, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from prettytable import PrettyTable
from tqdm import tqdm, trange

from utils import Utils


class Worker:
    """
    A worker in the annotating system, corresponding to the arm in the CMAB algorithm.
    """
    def __init__(self, id: int, true_f1: float, true_mv_f1: float, utils: Utils):
        self.id = id
        self.true_f1 = true_f1
        self.true_mv_f1 = true_mv_f1
        self.f1s = []
        self.annotation_num = 0
        self.utils = utils

    @property
    def average_f1(self):
        # This costs a lot of time, to be optimised.
        return np.mean(self.f1s)

    def annotate(self, sentence_ids: Union[list[int], int], update_f1: bool = True) -> int:
        """
        Annotate new sentences, and return f1 of those annotations.
        """
        if random.random() < self.true_f1:
            f1 = 1
        else:
            f1 = 0

        if update_f1:
            self.f1s.append(f1)

        self.annotation_num += len(sentence_ids)
        self.utils.update_unannotated_sentences(self.id, sentence_ids)
        return f1

    def __repr__(self):
        return f'{self.id}'


class Manager:
    """
    The annotating system manager, corresponding to the bandit player in the CMAB algorithm.
    """

    def __init__(self, workers: list[Worker], utils: Utils, selected_worker_num: int = 20, ucb_scale: float = 1.0,
                 allow_exhaustion: bool = True, evaluation_interval: int = 10):
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

        self.annotation_f1s: dict[int, list[float]] = {}


        self.selection_history = []
        self.regret_history = []
        self.utils = utils

        self.sum_empirical_best_f1s = sum(
            sorted(self.utils.worker2mean_f1.values(), reverse=True)[:selected_worker_num])

    def initialize(self) -> NoReturn:
        """
        Before normal rounds, let each worker annotate once for reference in later selections.
        """
        for worker in self.workers:
            self.t += 1
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            f1 = worker.annotate(sentence_ids)
            for sentence_id in sentence_ids:
                if sentence_id not in self.annotation_f1s.keys():
                    self.annotation_f1s[sentence_id] = []
                self.annotation_f1s[sentence_id].append(f1)

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
        for worker in selected_workers:
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num = 0
            f1 = worker.annotate(sentence_ids, update_f1=is_eval_step)
            for sentence_id in sentence_ids:
                if sentence_id not in self.annotation_f1s.keys():
                    self.annotation_f1s[sentence_id] = []
                self.annotation_f1s[sentence_id].append(f1)
                step_f1 += f1

        self.selection_history.append(selected_workers)
        if actual_worker_num < 20:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1)

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

    def evaluate(self, dump_results: bool = True) -> Tuple[float, int]:
        """
        Calculate classification scores and dump results.
        """

        all_f1s: list[float] = []
        for _, f1s in self.annotation_f1s.items():
            all_f1s.extend(f1s)
        f1 = float(np.mean(all_f1s))

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

        annotation_num: int = reduce(lambda num1, num2: num1 + num2, map(lambda worker: worker.annotation_num, self.workers))

        return f1, annotation_num

    def plot_regret(self):
        plt.plot(range(len(self.regret_history)), self.regret_history)
        plt.xlabel('Time step')
        plt.ylabel('Regret')
        plt.show()


class MVManager(Manager):
    """
    Use majority voting to decide when to access the expert annotations.
    """

    def __init__(self, workers: list[Worker], utils: Utils, mv_ratio: float = 0.4, **kwargs):
        super().__init__(workers, utils, **kwargs)
        self.annotation_workers: dict[int, list[Worker]] = {}

        self.expert_use_num = 0
        self.sent2expert_tag_ratio = {}

        self.sent2is_mv_tags_correct: dict[int, bool] = {}  # Sentence level correct
        self.sent2mv_tags_correct_ratio: dict[int, float] = {}  # Token level correct

        self.confidence_bounds: list[float] = []

        self.mv_ratio = mv_ratio

        self.expert_use_num = 0

    def initialize(self) -> NoReturn:
        """
        Before normal rounds, let each worker annotate once for reference in later selections.
        """
        for worker in self.workers:
            self.t += 1
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            f1 = worker.annotate(sentence_ids)
            for sentence_id in sentence_ids:
                if sentence_id not in self.annotation_f1s.keys():
                    self.annotation_f1s[sentence_id] = []
                    self.annotation_workers[sentence_id] = []
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)
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
                self.confidence_bounds.append(self.ucb_scale * np.sqrt(3 * np.log(self.t) / (2 * worker.annotation_num)))

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
        for worker in selected_workers:
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue

            actual_worker_num += 1
            f1 = worker.annotate(sentence_ids, update_f1=False)

            for sentence_id in sentence_ids:
                if sentence_id not in self.annotation_f1s.keys():
                    self.annotation_f1s[sentence_id] = []
                    self.annotation_workers[sentence_id] = []
                self.annotation_f1s[sentence_id].append(f1)
                self.annotation_workers[sentence_id].append(worker)

                # Check if all annotations on this sentence are done.
                if self.utils.allow_fake_annotations:
                    voter_num = self.utils.annotation_num_per_sentence
                else:
                    voter_num = min(len(self.utils.sent2workers[sentence_id]), self.utils.annotation_num_per_sentence)
                if len(self.annotation_workers[sentence_id]) == voter_num:
                    # Update f1 score records of workers.
                    if random.random() > self.mv_ratio:
                        for worker in self.annotation_workers[sentence_id]:
                            worker.f1s.append(1 if random.random() < worker.true_mv_f1 else 0)
                    else:
                        for worker in self.annotation_workers[sentence_id]:
                            worker.f1s.append(1 if random.random() < worker.true_f1 else 0)
                        self.expert_use_num += 1

                step_f1 += f1

        self.selection_history.append(selected_workers)
        if actual_worker_num < 20:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1)


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
        for worker in selected_workers:
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            f1 = worker.annotate(sentence_ids, update_f1=False)
            for sentence_id in sentence_ids:
                if sentence_id not in self.annotation_f1s.keys():
                    self.annotation_f1s[sentence_id] = []
                self.annotation_f1s[sentence_id].append(f1)
                step_f1 += f1

        self.selection_history.append(selected_workers)
        if actual_worker_num < 20:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1)


class RandomManager(Manager):
    """
    Always select the best workers according to the average F-1 on real dataset in each round.
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
        for worker in selected_workers:
            sentence_ids = self.utils.get_unannotated_sentence_ids(worker.id, num=1)
            # If this worker has no more annotations to do, forget about him in the following rounds.
            if not sentence_ids:
                if self.allow_exhaustion:
                    self.exhausted_workers.append(worker)
                else:
                    self.utils.restore_unannotated_sentences()
                continue
            actual_worker_num += 1
            f1 = worker.annotate(sentence_ids, update_f1=False)
            for sentence_id in sentence_ids:
                if sentence_id not in self.annotation_f1s.keys():
                    self.annotation_f1s[sentence_id] = []
                self.annotation_f1s[sentence_id].append(f1)
                step_f1 += f1

        self.selection_history.append(selected_workers)
        if actual_worker_num < 20:
            return
        self.regret_history.append(self.sum_empirical_best_f1s - step_f1)

def run_average(average_over: int = 100,
                num_steps: int = 10000,
                annotation_num_per_sentence: int = 1,
                allow_exhaustion: bool = True,
                evaluation_interval: int = 10,
                manager_type: str = 'normal',
                allow_fake_annotations: bool = False,
                metrics_type: str = 'span_exact',
                ucb_scale: float = 1.0,
                mv_ratio: float = 0.1) -> NoReturn:
    if manager_type == 'normal':
        manager_class = Manager
    elif manager_type == 'mv_expert':
        manager_class = MVManager
    elif manager_type == 'best':
        manager_class = BestManager
    elif manager_type == 'random':
        manager_class = RandomManager
    else:
        raise ValueError(f'Manager type should be \'normal\' or \'mv_expert\', but {manager_type} is given.')

    f1s = []
    confidence_bounds = []
    expert_use_nums = []

    annotation_num = 0
    for _ in trange(average_over, desc=f'Averaging over {average_over} trials: ', position=0, ncols=80):
        utils = Utils(annotation_num_per_sentence,
                      allow_fake_annotations=allow_fake_annotations,
                      metrics_type=metrics_type)
        workers = [
            Worker(worker_id, utils.worker2mean_f1[worker_id], utils.worker2mean_mv_f1[worker_id], utils)
            for worker_id in utils.worker_ids
        ]
        manager = manager_class(workers, utils,
                                allow_exhaustion=allow_exhaustion,
                                evaluation_interval=evaluation_interval,
                                ucb_scale=ucb_scale)
        if manager_class == MVManager:
            manager.mv_ratio = mv_ratio
        manager.run(num_steps=num_steps)
        f1, annotation_num = manager.evaluate(dump_results=False)

        f1s.append(f1 * 100)

        if manager_class == MVManager:
            expert_use_nums.append(manager.expert_use_num)
            confidence_bounds.extend(manager.confidence_bounds)

    print(f'Evaluation over {average_over} trials.')
    print(f'Produced {annotation_num} annotations in total.\n')

    print(f'Mean F1: {np.mean(f1s):.02f}\n')

    if manager_class == MVManager:
        print(f'Expert usage: {np.mean(expert_use_nums)} / {len(utils.sent2silver_tags)}'
              f' = {np.mean(expert_use_nums) / len(utils.sent2silver_tags):.02%}')

    # if manager_class == MVManager:
    #     print(f'Confidence bound stats:')
    #     print(f'Min: {min(confidence_bounds)}')
    #     print(f'Max: {max(confidence_bounds)}')
    #     print(f'Avg: {np.mean(confidence_bounds)}\n')


def run_once(num_steps: int = 10000,
             annotation_num_per_sentence: int = 1,
             allow_exhaustion: bool = True,
             evaluation_interval=10,
             manager_type: str = 'normal',
             allow_fake_annotations: bool = False,
             metrics_type: str = 'span_exact',
             ucb_scale: float = 1.0,
             mv_ratio: float = 0.4) -> NoReturn:
    global utils
    if manager_type == 'normal':
        manager_class = Manager
    elif manager_type == 'mv_expert':
        manager_class = MVManager
    elif manager_type == 'best':
        manager_class = BestManager
    elif manager_type == 'random':
        manager_class = RandomManager
    else:
        raise ValueError(f'Manager type should be \'normal\' or \'mv_expert\', but {manager_type} is given.')
    utils = Utils(annotation_num_per_sentence=annotation_num_per_sentence,
                  allow_fake_annotations=allow_fake_annotations,
                  metrics_type=metrics_type)
    workers = [
        Worker(worker_id, utils.worker2mean_f1[worker_id], utils.worker2mean_mv_f1[worker_id], utils)
        for worker_id in utils.worker_ids
    ]
    manager = manager_class(workers, utils,
                            allow_exhaustion=allow_exhaustion,
                            evaluation_interval=evaluation_interval,
                            ucb_scale=ucb_scale)
    if manager_class == MVManager:
        manager.mv_ratio = mv_ratio
    manager.run(num_steps=num_steps)
    f1, annotation_num = manager.evaluate(dump_results=False)
    manager.plot_regret()
    # utils.export_selections(manager.workers)
    print(f'\nProduced {annotation_num} annotations in total.')
    print(f'{annotation_num / len(utils.sent2workers):.02f} annotations per sentence on average.\n')

    print(f'Mean F1: {f1}\n')

    # if manager_class == MVManager:
    #     print(f'Confidence bound stats:')
    #     print(f'Min: {min(manager.confidence_bounds)}')
    #     print(f'Max: {max(manager.confidence_bounds)}')
    #     print(f'Avg: {np.mean(manager.confidence_bounds)}\n')


if __name__ == '__main__':
    # run_once(num_steps=5000,
    #          annotation_num_per_sentence=4,
    #          allow_exhaustion=True,
    #          evaluation_interval=0,
    #          allow_fake_annotations=True,
    #          manager_type='mv_expert',
    #          metrics_type='span_proportional',
    #          ucb_scale= 1.0,
    #          mv_ratio= 0.1
    # )

    run_average(average_over=10,
                num_steps=5000,
                annotation_num_per_sentence=4,
                allow_exhaustion=True,
                evaluation_interval=0,
                allow_fake_annotations=True,
                manager_type='mv_expert',
                metrics_type='span_proportional',
                ucb_scale=1.0,
                mv_ratio=0.4
    )

    # 8047, 1.0 = 469 rounds
    # 15046, 1.87 = 819 rounds
    # 20805, 2.59 = 1107 rounds
