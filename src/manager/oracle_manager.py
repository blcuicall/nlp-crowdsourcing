import random

from src.data_structure import Annotation, Worker
from src.dataset import Dataset
from src.manager.manager import Parameters, Manager


class OracleManager(Manager):
    def __init__(self, dataset: Dataset, parameters: Parameters):
        super().__init__(dataset, parameters)

        self.best_workers: list[Worker] = [worker for worker, score in
                                           list(sorted(self.dataset.worker2avg_score.items(),
                                                       key=lambda item: item[1],
                                                       reverse=True)
                                                )[:self.parameters.num_workers_in_step]]

    def select_workers(self) -> list[Worker]:
        """
        Always select the best workers.
        """
        selected_workers: list[Worker] = list(self.best_workers)
        return selected_workers
