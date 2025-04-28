import random
from collections import defaultdict
from copy import copy

from src.agreement import fleiss_kappa
from src.data_augmentor import YACLCDataAugmentor
from src.data_structure import Sentence, Annotation


def expert_evaluate(annos: list[Annotation]) -> dict[Annotation, float]:
    """
    Simply evaluate the annotations w.r.t. expert annotations.
    These scores are already calculated and cached in the dataset.
    """
    return {anno: anno.score for anno in annos}


def mv_evaluate(annos: list[Annotation]) -> dict[Annotation, float]:
    """
    Evaluate the annotations w.r.t. the majority vote of them.
    These scores are not cached in the dataset.
    Since the combinations of annotations on a sentence are nearly infinite,
    we have to calculate the majority vote score on the fly.
    """
    # The following commented code is an unfinished approach to vote on edit level.
    # edit_seqs: list[list] = [list(anno.edit_seq) for anno in annos]
    # max_edit_num = max([len(edit_seq) for edit_seq in edit_seqs])
    # for edit_seq in edit_seqs:
    #     while len(edit_seq) < max_edit_num:
    #         edit_seq.append('O')

    # mv_seq: list = get_mv_seq(edit_seqs)
    # while 'O' in mv_seq:
    #     mv_seq.remove('O')
    # mv_anno: Annotation = copy(annos[0])
    # YACLCDataAugmentor.apply_edits(tuple(mv_seq), mv_anno.sent.content, )

    # The following code is an approach to vote on whole edit sequence level.
    votes = defaultdict(int)
    for anno in annos:
        votes[str(anno.content)] += 1

    content_with_same_votes = []
    for i, content_item in enumerate(list(sorted(votes.items(), key=lambda x: x[1], reverse=True))):
        if i == 0:
            content_with_same_votes.append(content_item)
            continue

        vote_num = content_item[1]
        if vote_num == content_with_same_votes[0][1]:
            content_with_same_votes.append(content_item)
        else:
            break

    mv_content = random.choice(content_with_same_votes)[0]

    mv_anno: Annotation = None
    for anno in annos:
        if str(anno.content) == mv_content:
            mv_anno = anno
            break

    anno2score = YACLCDataAugmentor.evaluate_annos(annos, [mv_anno for _ in annos])
    return anno2score, mv_anno


def mv_expert_evaluate(annos: list[Annotation], kappa_threshold: float = 0.2) -> tuple[dict[Annotation, float], str]:
    """
    Evaluate the annotations w.r.t. the majority vote of them or the expert annotation.
    This depends on whether the Fleiss' Kappa score is higher than the preset threshold.
    """
    annos_formatted: list[list[str]] = [[str(edit) for edit in anno.edit_seq] for anno in annos]
    max_edit_num = max([len(edit_seq) for edit_seq in annos_formatted])
    for edit_seq in annos_formatted:
        while len(edit_seq) < max_edit_num:
            edit_seq.append('O')
    agreement = fleiss_kappa(annos_formatted)
    if agreement > kappa_threshold:
        anno2score, mv_anno = mv_evaluate(annos)
        return anno2score, 'mv', mv_anno
    else:
        anno2score = expert_evaluate(annos)
        return anno2score, 'expert', random.choice(list(anno2score.keys()))


def get_mv_seq(edit_seqs: list[list]) -> list:
    """
    Get the majority vote sequence of edits.
    """
    votes = [defaultdict(int) for _ in range(len(edit_seqs[0]))]

    # pos2votes: dict[int, dict[str, int]] = {}

    for edit_seq in edit_seqs:
        for i, edit in enumerate(edit_seq):
            votes[i][edit] += 1

    mv_seq = [sorted(vote.items(), key=lambda x: x[1], reverse=True)[0][0] for vote in votes]

    return mv_seq
