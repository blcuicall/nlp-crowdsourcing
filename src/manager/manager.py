import random
from typing import NoReturn

import numpy as np
from tqdm import tqdm

from src.manager.evaluation import expert_evaluate
from src.data_structure import Annotation, Worker, Sentence
from src.dataset import Dataset


class Parameters:

    def __init__(self, num_workers_in_step: int = 20,
                 num_anno_per_sent: int = 10,
                 kappa_threshold: float = 0.2,
                 ucb_scale: float = 1.0,
                 sample_num: int = 100,
                 epsilon: float = 0.5,
                 evaluation_type: str = 'expert'):
        self.num_workers_in_step: int = num_workers_in_step
        self.num_anno_per_sent: int = num_anno_per_sent
        self.kappa_threshold: float = kappa_threshold

        self.ucb_scale: float = ucb_scale
        self.epsilon: float = epsilon
        self.sample_num: int = sample_num
        self.evaluation_type: str = evaluation_type


class Manager:
    def __init__(self, dataset: Dataset, parameters: Parameters):
        self.dataset: Dataset = dataset
        self.parameters: Parameters = parameters

        self.unannotated_sents: list[Sentence] = list(self.dataset.sents)
        self.sent2selected_annos: dict[Sentence, list[Annotation]] = {
            sent: [] for sent in self.dataset.sents
        }

        self.worker2current_scores: dict[Worker, list[float]] = {
            worker: [] for worker in self.dataset.workers
        }

        self.t = 0

        self.regret_history: list[float] = []
        self.best_step_score: float = self.dataset.best_step_score(self.parameters.num_workers_in_step)

    @property
    def worker2current_anno_nums(self) -> dict[Worker, int]:
        return {worker: len(scores) for worker, scores in self.worker2current_scores.items()}

    def select_workers(self) -> list[Worker]:
        """
        Randomly select workers.
        """
        selected_workers: list[Worker] = random.sample(self.dataset.workers, self.parameters.num_workers_in_step)
        return selected_workers

    def select_sentences(self) -> list[Sentence]:
        """
        Randomly select sentences to be annotated in a step.
        """
        sample_size = int(self.parameters.num_workers_in_step / self.parameters.num_anno_per_sent)
        if sample_size > len(self.unannotated_sents):
            sample_size = len(self.unannotated_sents)
        selected_sents: list[Sentence] = random.sample(self.unannotated_sents, sample_size)
        return selected_sents

    def step(self):
        # Select workers according to the manager's policy.
        selected_workers: list[Worker] = self.select_workers()

        # Select sentences to be annotated in this step.
        selected_sents2annos: dict[Sentence, list[Annotation]] = {
            sent: [] for sent in self.select_sentences()
        }

        # Select annotations for each sentence.
        self.select_annotations(selected_workers, selected_sents2annos)

        # Evaluate the selected annotations.
        self.evaluate_selected_annotations(selected_sents2annos)

        # Update regret history.
        self.update_regret_history(selected_sents2annos)

        # Update runtime parameters, e.g., time.
        self.update_runtime_params()

    def select_annotations(self,
                           selected_workers: list[Worker],
                           selected_sents2annos: dict[Sentence, list[Annotation]]) -> NoReturn:
        for sent in selected_sents2annos.keys():
            while len(selected_sents2annos[sent]) < self.parameters.num_anno_per_sent:
                worker = selected_workers.pop()

                selected_anno: Annotation = None
                for anno in self.dataset.sent2annos[sent]:
                    if anno.worker == worker:
                        selected_anno = anno

                selected_sents2annos[sent].append(selected_anno)
                self.sent2selected_annos[sent].append(selected_anno)

    def evaluate_selected_annotations(self, selected_sents2annos: dict[Sentence, list[Annotation]]) -> NoReturn:
        """
        Simply evaluate the annotations w.r.t. expert annotations.
        These scores are already calculated and cached in the dataset.
        """
        for sent, annos in selected_sents2annos.items():
            anno2score = expert_evaluate(annos)
            for anno, score in anno2score.items():
                self.worker2current_scores[anno.worker].append(score)

    def update_regret_history(self, selected_sents2annos: dict[Sentence, list[Annotation]]) -> NoReturn:
        step_score = 0
        for sent, annos in selected_sents2annos.items():
            self.unannotated_sents.remove(sent)
            step_score += sum(anno.score for anno in annos)
        self.regret_history.append(self.best_step_score - step_score)

    def update_runtime_params(self) -> NoReturn:
        self.t += 1

    def run(self, num_step=-1):
        current_step = 0
        # with tqdm(total=len(self.unannotated_sents), desc='Current run', leave=False) as progress_bar:
        while self.unannotated_sents:
            if current_step >= num_step != -1:
                break
            current_step += 1
            self.step()
                # progress_bar.update(int(self.parameters.num_workers_in_step / self.parameters.num_anno_per_sent))

    def get_results(self) -> tuple[float, list[float]]:
        scores = []
        for sent, annos in self.sent2selected_annos.items():
            for anno in annos:
                scores.append(anno.score)
        return np.average(scores), self.regret_history


