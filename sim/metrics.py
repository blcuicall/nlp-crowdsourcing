from typing import Optional, Set, Tuple, Union, Any
from sklearn.metrics import f1_score as skl_f1_score, recall_score as skl_recall_score, \
    precision_score as skl_precision_score
from seqeval.metrics import f1_score as seq_f1_score, recall_score as seq_recall_score, \
    precision_score as seq_precision_score

from torch import LongTensor
import torch
from typing import Dict, Set, Tuple
from argparse import Namespace
from collections import OrderedDict
from itertools import chain


class Metrics:

    def __init__(self, metrics_type: str = 'span_exact'):
        self.metrics_type = metrics_type
        if self.metrics_type not in ['span_exact', 'span_proportional', 'token']:
            raise ValueError(
                f'Wrong metrics level given: {self.metrics_type}. Should be \'span_exact\', \'span_proportional\' or \'token\'')

        self.tag2index = {
            'O': 0,
            'B-POS': 1,
            'I-POS': 2,
            'B-NEG': 3,
            'I-NEG': 4,
        }

        self.span_proportional_match = Proportional(o_id=0, token_to_id=self.tag2index, output_detail=True)

    def f1_score(self, y_true: list[list[str]], y_pred: list[list[str]], metrics_type: Optional[str] = None):
        if not metrics_type:
            metrics_type = self.metrics_type
        if metrics_type == 'span_exact':
            f1 = seq_f1_score(y_true, y_pred, average='macro')
        elif metrics_type == 'span_proportional':
            f1 = self.span_proportional_scores(y_true, y_pred)['F1']
        elif metrics_type == 'token':
            f1 = skl_f1_score(self.to_skl_format(y_true), self.to_skl_format(y_pred), average='macro')
        else:
            f1 = None
        return f1

    def precision_score(self, y_true: list[list[str]], y_pred: list[list[str]], metrics_type: Optional[str] = None):
        if not metrics_type:
            metrics_type = self.metrics_type
        if metrics_type == 'span_exact':
            precision = seq_precision_score(y_true, y_pred, average='macro')
        elif metrics_type == 'span_proportional':
            precision = self.span_proportional_scores(y_true, y_pred)['precision']
        elif metrics_type == 'token':
            precision = skl_precision_score(self.to_skl_format(y_true), self.to_skl_format(y_pred), average='macro')
        else:
            precision = None
        return precision

    def recall_score(self, y_true: list[list[str]], y_pred: list[list[str]], metrics_type: Optional[str] = None):
        if not metrics_type:
            metrics_type = self.metrics_type
        if metrics_type == 'span_exact':
            recall = seq_recall_score(y_true, y_pred, average='macro')
        elif metrics_type == 'span_proportional':
            recall = self.span_proportional_scores(y_true, y_pred)['recall']
        elif metrics_type == 'token':
            recall = skl_recall_score(self.to_skl_format(y_true), self.to_skl_format(y_pred), average='macro')
        else:
            recall = None
        return recall

    def to_skl_format(self, annotations: list[list[str]]) -> list[int]:
        skl_annotations = []
        for annotation in annotations:
            for tag in annotation:
                skl_annotations.append(self.tag2index[tag])
        return skl_annotations

    def to_oei_format(self, annotations: list[list[str]]) -> torch.LongTensor:
        max_length = max(map(lambda annotation: len(annotation), annotations))
        oei_annotations = []
        for annotation in annotations:
            oei_annotation = []
            for tag in annotation:
                oei_annotation.append(self.tag2index[tag])
            # Padding
            while len(oei_annotation) < max_length:
                oei_annotation.append(self.tag2index['O'])
            oei_annotations.append(oei_annotation)

        return torch.LongTensor(oei_annotations)

    def span_proportional_scores(self, y_true: list[list[str]], y_pred: list[list[str]]) -> dict[str, float]:
        y_true_tensor = self.to_oei_format(y_true)
        y_pred_tensor = self.to_oei_format(y_pred)
        y_lengths_tensor = torch.LongTensor([[len(annotation)] for annotation in y_pred])
        return self.span_proportional_match(y_pred_tensor, y_true_tensor, y_lengths_tensor)


# Code below is copied from the OEI repository

def a_better_than_b(a, b):
    for k, v in a.items():
        if v > b[k]:
            return True
        elif v < b[k]:
            return False
    return False


def namespace_add(a, b):
    return Namespace(**{k: a.__dict__[k] + b.__dict__[k] for k in a.__dict__})


class OEIMetric:
    """
    A very general abstract class representing a metric which can be accumulated.
    """

    def __init__(self):
        self.counter = self.counter_factory()
        self.best = None

    def is_best(self, metric: Dict) -> bool:
        """
        根据key的顺序比较metric，在前者优先，默认数值越大越好。
        """
        if self.best is None or a_better_than_b(metric, self.best):
            self.best = metric
            return True
        return False

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> Dict:
        """
        每个batch调用，更新counter，计算当前batch的分数并返回。
        """
        raise NotImplementedError

    def get_metric(self, counter=None, reset=False) -> Dict:
        """
        用counter计算出metric。
        """
        raise NotImplementedError

    @staticmethod
    def counter_factory(**kwargs) -> Namespace:
        raise NotImplementedError

    @staticmethod
    def metric_factory(**kwargs) -> Dict:
        """
        注意按重要性排列参数。
        """
        raise NotImplementedError


class TaggingOEIMetric(OEIMetric):
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> Dict:
        batch = self.counter_factory()

        mask = (gold_labels != self.ignore_index).long() * mask  # 只看标注
        batch.total = mask.sum().item()
        batch.positive = ((predictions != self.ignore_index).long() * mask).sum().item()
        batch.correct = ((predictions == gold_labels).long() * mask).sum().item()

        self.counter = namespace_add(self.counter, batch)

        return self.get_metric(batch)

    @staticmethod
    def counter_factory(total=0, positive=0, correct=.0) -> Namespace:
        return Namespace(total=total, positive=positive, correct=correct)

    @staticmethod
    def metric_factory(f1=.0, recall=.0, precision=.0) -> Dict:
        return dict(F1=f1, recall=recall, precision=precision)

    def get_metric(self, counter=None, reset=False) -> Dict:
        c = counter or self.counter
        # De facto micro average
        total, correct, positive = c.total, c.correct, c.positive
        recall = 0 if total == 0 else correct / total
        precision = 0 if positive == 0 else correct / positive

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if reset:
            self.counter = self.counter_factory()

        return self.metric_factory(f1, recall, precision)


class ExactMatch(TaggingOEIMetric):
    def __init__(self, o_id, token_to_id, ouput_class=False, output_detail=False):
        super().__init__(o_id)
        self.o_id = o_id
        self.id_to_label = dict()  # map[i_x] = x
        self.bi_map = dict()  # map[b_x] = i_x
        for label, index in token_to_id.items():
            if label.startswith('B-'):
                self.bi_map[label[2:]] = index
        for label, index in token_to_id.items():
            if label.startswith('I-'):
                b_id = self.bi_map.pop(label[2:])
                self.bi_map[b_id] = index
                self.id_to_label[index] = label[2:]

        self.label_counter = {k: self.counter_factory() for k in self.id_to_label}
        self.ouput_class = ouput_class
        self.output_detail = output_detail
        self.data_info = dict()

    def __call__(self,
                 predictions: LongTensor,
                 gold_labels: LongTensor,
                 lengths: LongTensor) -> dict:
        batch = self.counter_factory()

        for prediction, gold, length in zip(predictions, gold_labels, lengths):
            predict_entities = self.get_entities(prediction.tolist()[:length])
            gold_entities = self.get_entities(gold.tolist()[:length])
            correct_entities = self.get_correct(predict_entities, gold_entities)

            for e in gold_entities:
                self.label_counter[e[2]].total += 1
                batch.total += 1
            for e in predict_entities:
                self.label_counter[e[2]].positive += 1
                batch.positive += 1
            for e in correct_entities:
                self.label_counter[e[2]].correct += e[3]
                batch.correct += e[3]

        self.counter = namespace_add(self.counter, batch)

        return self.get_metric(batch)

    def get_entities(self, labels) -> set[tuple[Union[int, Any], ...]]:
        entities, one = set(), None
        for i, label in enumerate(chain(labels, [self.o_id])):
            if one:
                if label == one[2]:  # I-x
                    one[1] = i
                    continue
                else:
                    entities.add(tuple(one))
                    one = None
            if label in self.bi_map:  # B-x
                one = [i, i, self.bi_map[label]]  # start, end, I-x
        return entities

    @staticmethod
    def get_correct(predict_entities, gold_entities):
        correct_entities = predict_entities & gold_entities
        correct_entities = {tuple(chain(e, [1])) for e in correct_entities}
        return correct_entities

    def get_metric(self, counter=None, reset=False) -> dict:
        if not reset:
            return super().get_metric(counter)
        if not self.ouput_class:
            return super().get_metric(reset=True)

        key_list = ['F1', 'precision', 'recall'] if self.output_detail else ['F1']

        metrics = dict(main=super().get_metric(reset=True))
        for k, counter in self.label_counter.items():
            self.data_info[self.id_to_label[k]] = counter.total
            metrics[k] = super().get_metric(counter)
            self.label_counter[k] = self.counter_factory()

        metric_with_prefix = OrderedDict()
        for prefix in chain(['main'], self.label_counter.keys()):
            for k in key_list:
                prefix_str = self.id_to_label[prefix] if isinstance(prefix, int) else prefix
                metric_with_prefix[f"{prefix_str}_{k}"] = metrics[prefix][k]

        return metric_with_prefix


class Binary(ExactMatch):
    @staticmethod
    def get_correct(predict_entities, gold_entities):
        correct_entities = set()
        for e in predict_entities:
            for g in gold_entities:
                if e[2] != g[2]:
                    continue
                if e[0] > g[1]:
                    continue
                if e[1] < g[0]:
                    continue
                correct_entities.add(tuple(chain(e, [1])))
        return correct_entities


class Proportional(ExactMatch):
    @staticmethod
    def get_correct(predict_entities, gold_entities):
        correct_entities = set()
        for e in predict_entities:
            for g in gold_entities:
                if e[2] != g[2]:
                    continue
                if e[0] > g[1]:
                    continue
                if e[1] < g[0]:
                    continue
                p = (min(e[1], g[1]) - max(e[0], g[0]) + 1) / (g[1] - g[0] + 1)
                correct_entities.add(tuple(chain(e, [p])))
        return correct_entities


if __name__ == '__main__':
    y_pred = [
        ['O', 'O', 'O', 'O', 'O', 'B-POS', 'I-POS', 'I-POS', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'I-NEG', 'I-NEG',
         'I-NEG', 'I-NEG', 'I-NEG'],
        ['O', 'O', 'O', 'O', 'O', 'B-POS', 'I-POS', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NEG', 'I-NEG']
    ]
    y_true = [
        ['O', 'O', 'O', 'O', 'O', 'B-POS', 'I-POS', 'I-POS', 'I-POS', 'I-POS', 'O', 'O', 'O', 'O', 'O', 'B-NEG',
         'I-NEG', 'I-NEG', 'I-NEG'],
        ['O', 'O', 'O', 'O', 'O', 'B-POS', 'I-POS', 'I-POS', 'I-POS', 'I-POS', 'O', 'O', 'O', 'O', 'O', 'B-NEG',
         'I-NEG', 'I-NEG', 'I-NEG']
    ]

    metrics = Metrics('span_proportional')
    print(metrics.f1_score(y_true, y_pred))
