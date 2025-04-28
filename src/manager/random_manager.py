import random

from src.data_structure import Annotation, Worker
from src.dataset import Dataset
from src.manager.manager import Parameters, Manager


class RandomManager(Manager):
    """
    Just a renamed duplicate of the Manager class.
    """
    def __init__(self, dataset: Dataset, parameters: Parameters):
        super().__init__(dataset, parameters)
