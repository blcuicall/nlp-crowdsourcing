import json
import random
from typing import NoReturn

import numpy as np

from src.manager.evaluation import expert_evaluate, mv_evaluate, mv_expert_evaluate
from src.data_structure import Annotation, Worker, Sentence
from src.dataset import Dataset
from src.manager import Parameters, Manager


class ThompsonSamplingManager(Manager):
    def __init__(self, dataset: Dataset, parameters: Parameters):
        super().__init__(dataset, parameters)

        self.worker2alpha: dict[Worker, float] = {worker: 1.0 for worker in self.dataset.workers}
        self.worker2beta: dict[Worker, float] = {worker: 1.0 for worker in self.dataset.workers}

        self.expert_evaluation_usage: int = 0

        self.mv_hits_on_expert: int = 0

        with open('out/gold_anno.json', 'r') as reader:
            data = json.load(reader)
            self.sent_id2gold_anno_content: dict[int, list[str]] = {int(k): v for k, v in data.items()}

    def select_workers(self) -> list[Worker]:
        """
        Select workers according to the Thompson sampling policy.
        """

        worker2sample_reward: dict[Worker, float] = {}
        for worker in self.worker2alpha.keys():
            worker2sample_reward[worker] = float(np.sum(np.random.beta(a=self.worker2alpha[worker],
                                                                       b=self.worker2beta[worker],
                                                                       size=self.parameters.sample_num)))
        worker2sample_reward = dict(sorted(worker2sample_reward.items(), key=lambda item: item[1], reverse=True))

        selected_workers = list(worker2sample_reward.keys())[:self.parameters.num_workers_in_step]

        return selected_workers

    def evaluate_selected_annotations(self, selected_sents2annos: dict[Sentence, list[Annotation]]) -> NoReturn:
        for sent, annos in selected_sents2annos.items():
            if self.parameters.evaluation_type == 'expert':
                anno2score: dict[Annotation, float] = expert_evaluate(annos)
            elif self.parameters.evaluation_type == 'mv':
                anno2score: dict[Annotation, float] = mv_evaluate(annos)
            elif self.parameters.evaluation_type == 'mv_expert':
                anno2score, actual_evaluation_type, mv_anno = mv_expert_evaluate(annos, self.parameters.kappa_threshold)
                if actual_evaluation_type == 'expert':
                    self.expert_evaluation_usage += 1
                elif mv_anno.content == self.sent_id2gold_anno_content[sent.id]:
                    self.mv_hits_on_expert += 1
            else:
                raise ValueError(f'Unknown evaluation type: {self.parameters.evaluation_type}')

            for anno, score in anno2score.items():
                self.worker2current_scores[anno.worker].append(score)
                self.worker2alpha[anno.worker] += score
                self.worker2beta[anno.worker] += 1 - score

    def update_runtime_params(self) -> NoReturn:
        super().update_runtime_params()
