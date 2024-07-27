import json
import random
from typing import List, Union, NoReturn, Tuple, Dict

from metrics import Metrics

import warnings

warnings.filterwarnings('error')


class Utils:

    def __init__(self, annotation_num_per_sentence=1,
                 allow_fake_annotations=False,
                 metrics_type='span_exact'):
        self.annotation_num_per_sentence = annotation_num_per_sentence
        if metrics_type not in ['span_exact', 'span_proportional', 'token']:
            raise ValueError(
                f'Wrong metrics level given: {metrics_type}. Should be \'span_exact\', \'span_exact\' or \'token\'')

        self.metrics = Metrics(metrics_type=metrics_type)
        self.allow_fake_annotations = allow_fake_annotations

        self.tag2index = {
            'O': 0,
            'B-POS': 1,
            'I-POS': 2,
            'B-NEG': 3,
            'I-NEG': 4,
        }

        with open('data/train.json', 'r') as data_file:
            self.data = json.load(data_file)

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

        # Mean F-1 of workers
        self.worker2mean_f1 = {}
        for worker, sents in self.worker2sents.items():
            worker_annotations = []
            mv_annotations = []
            for sentence in sents:
                worker_annotations.append(self.sent2all_tags[sentence][worker])
                mv_annotations.append(self.sent2silver_tags[sentence])
            self.worker2mean_f1[worker] = self.metrics.f1_score(mv_annotations, worker_annotations)

        # Key: worker id
        # Value: ids of sentences annotated by this user in the train data but not yet annotated in the CMAB process
        self.worker2unannotated_sents: dict[int, list[int]] = {}
        for sent, worker_ids in self.sent2workers.items():
            for worker_id in worker_ids:
                if worker_id not in self.worker2unannotated_sents.keys():
                    self.worker2unannotated_sents[worker_id] = []
                self.worker2unannotated_sents[worker_id].append(sent)
        if allow_fake_annotations:
            for worker_id in self.worker_ids:
                for sent_id in self.sent2workers:
                    if sent_id not in self.worker2unannotated_sents[worker_id]:
                        self.worker2unannotated_sents[worker_id].append(sent_id)
        # If one sentence could have multiple annotations, duplicate workers' unannotated_sents
        for worker, sents in self.worker2unannotated_sents.items():
            multiplied_sents = []
            for sent in sents:
                multiplied_sents.extend([sent] * annotation_num_per_sentence)
            self.worker2unannotated_sents[worker] = multiplied_sents

        self.expert_use_num = 0
        self.sent2expert_tag_ratio = {}
        self.sent2mv_tags: dict[int, list[str]] = {}
        for sent_id in self.sent2all_tags.keys():
            self.sent2mv_tags[sent_id] = self.get_mv_expert_tags_from_original_dataset(sent_id)

        self.worker2mean_mv_f1: dict[int, float] = {}
        for worker, sents in self.worker2sents.items():
            worker_annotations = []
            mv_annotations = []
            for sent_id in sents:
                worker_annotations.append(self.sent2all_tags[sent_id][worker])
                mv_annotations.append(self.sent2mv_tags[sent_id])
            self.worker2mean_mv_f1[worker] = self.metrics.f1_score(mv_annotations, worker_annotations)

    def get_mv_expert_tags_from_original_dataset(self, sentence_id):
        mv_expert_tags = []
        votes = [{} for _ in self.sent2silver_tags[sentence_id]]
        for _, annotation in self.sent2all_tags[sentence_id].items():
            for i, tag in enumerate(annotation):
                if tag not in votes[i].keys():
                    votes[i][tag] = 0
                votes[i][tag] += 1

        # Select tags according to votes.
        if self.allow_fake_annotations:
            voter_num = self.annotation_num_per_sentence
        else:
            voter_num = min(len(self.sent2workers[sentence_id]), self.annotation_num_per_sentence)
        self.sent2expert_tag_ratio[sentence_id] = 0
        for i, vote in enumerate(votes):
            sorted_vote = {k: v for k, v in sorted(vote.items(), key=lambda item: item[1], reverse=True)}
            if list(sorted_vote.items())[0][1] / voter_num >= 0 and voter_num != 1:
                mv_expert_tags.append(list(sorted_vote.items())[0][0])
            else:
                mv_expert_tags.append(self.sent2silver_tags[sentence_id][i])
                self.sent2expert_tag_ratio[sentence_id] += 1
        if self.sent2expert_tag_ratio[sentence_id] > 0:
            self.expert_use_num += 1
        self.sent2expert_tag_ratio[sentence_id] /= len(self.sent2silver_tags[sentence_id])

        # Adjust invalid tags. E.g., 'O' followed by 'I-POS'.
        for i, tag in enumerate(mv_expert_tags):
            if i == 0:
                if 'I' in tag:
                    mv_expert_tags[i] = tag.replace('I', 'B')
                continue
            elif 'O' == mv_expert_tags[i - 1] and 'I' in tag:
                mv_expert_tags[i] = tag.replace('I', 'B')
            elif 'B-POS' == mv_expert_tags[i - 1] and 'B-POS' == tag:
                mv_expert_tags[i] = tag.replace('B', 'I')
            elif 'B-NEG' == mv_expert_tags[i - 1] and 'B-NEG' == tag:
                mv_expert_tags[i] = tag.replace('B', 'I')

        return mv_expert_tags

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
                    while sentence_id in self.worker2unannotated_sents[worker_id]:
                        self.worker2unannotated_sents[worker_id].remove(sentence_id)

    def restore_unannotated_sentences(self):
        for sent, worker_ids in self.sent2workers.items():
            for worker_id in worker_ids:
                if worker_id not in self.worker2unannotated_sents.keys():
                    self.worker2unannotated_sents[worker_id] = []
                self.worker2unannotated_sents[worker_id].append(sent)
        if self.allow_fake_annotations:
            for worker_id in self.worker_ids:
                for sent_id in self.sent2workers:
                    if sent_id not in self.worker2unannotated_sents[worker_id]:
                        self.worker2unannotated_sents[worker_id].append(sent_id)
        # If one sentence could have multiple annotations, duplicate workers' unannotated_sents
        for worker, sents in self.worker2unannotated_sents.items():
            multiplied_sents = []
            for sent in sents:
                multiplied_sents.extend([sent] * self.annotation_num_per_sentence)
            self.worker2unannotated_sents[worker] = multiplied_sents
        # self.worker2unannotated_sents: dict[int, list[int]] = {}
        # for sent, worker_ids in self.sent2workers.items():
        #     for worker_id in worker_ids:
        #         if worker_id not in self.worker2unannotated_sents.keys():
        #             self.worker2unannotated_sents[worker_id] = []
        #         self.worker2unannotated_sents[worker_id].append(sent)
        # # If one sentence could have multiple annotations, duplicate workers' unannotated_sents
        # self.worker2unannotated_sents = {
        #     worker: sents * self.annotation_num_per_sentence
        #     for worker, sents in self.worker2unannotated_sents.items()
        # }

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


if __name__ == '__main__':
    utils = Utils()

    sent2eval_type = {}
    sent2text = {}
    with open('out/annotations/oei/KappaAggManager-span_exact-aps=4-kt=0.4-f69.67-m69.67.json', 'r') as reader:
        data = json.load(reader)
    for line in data:
        sent2eval_type[line['id']] = line['reviewer']
        sent2text[line['id']] = line['text']

    sent2gold_line = {}
    with open('data/train-max.json', 'r') as reader:
        data = json.load(reader)
    for line in data:
        sent_data = {
            'id': line['id'],
            'text': line['text'],
            'annotations': line['annotations'],
            'reviewer': 1,
            'bestUsers': [1]
        }
        for span in sent_data['annotations']:
            span['user'] = 1
        sent2gold_line[line['id']] = sent_data

    dump_data = []
    for sent_id, tags in utils.sent2mv_tags.items():
        if sent2eval_type[sent_id] == 1: # Expert
            dump_data.append(sent2gold_line[sent_id])
            continue

        sent_data = {
            'id': sent_id,
            'text': sent2text[sent_id],
            'annotations': [],
            'reviewer': sent2eval_type[sent_id],
            'bestUsers': [2]
        }

        span = {
            "label": None,
            "start_offset": -1,
            "end_offset": -1,
            "user": 2
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
                    "user": 2
                }
                is_in_span = False
            elif is_in_span and tag.startswith('B-'):
                sent_data['annotations'].append(span)
                span = {
                    "label": tag.removeprefix('B-'),
                    "start_offset": i,
                    "end_offset": i + 1,
                    "user": 2
                }
            else:
                print(tags)
                raise ValueError(f'Invalid annotation found, sent: {sent_id}, worker: 2')
        dump_data.append(sent_data)

    with open(f'out/annotations/oei/train-exact-45exp-55mv.json', 'w') as out:
        out.write('[')
        for index, line in enumerate(dump_data):
            out.write(json.dumps(line, ensure_ascii=False))
            if index != len(dump_data) - 1:
                out.write(',\n')
            else:
                out.write(']')

