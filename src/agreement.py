from nltk.metrics import agreement


def fleiss_kappa(worker_tags: list[list[str]]) -> float:
    """
    Calculate the Fleiss' Kappa score of several(>=2) workers' annotations on the same sequence.
    Example of worker_tags:
    [
        ['B-POS', 'I-POS', 'O'    ],  # worker 1
        ['B-POS', 'O',     'O'    ],  # worker 2
        ['B-POS', 'I-POS', 'I-POS'],  # worker 3
    ]
    """
    assert len(worker_tags) >= 2

    # Check if all workers give the same (including no) annotation spans on the sequence.
    if worker_tags.count(worker_tags[0]) == len(worker_tags):
        return 1.0

    # The NLTK implementation.
    data = []
    for worker_idx, tags in enumerate(worker_tags):
        for tag_idx, tag in enumerate(tags):
            data.append((worker_idx, tag_idx, tag))
    task = agreement.AnnotationTask(data=data)
    return task.multi_kappa()


if __name__ == '__main__':
    # Running example.

    # data = [
    #     ['B-POS', 'I-POS', 'O'    ],  # worker 1
    #     ['B-POS', 'O',     'O'    ],  # worker 2
    #     ['B-POS', 'I-POS', 'I-POS'],  # worker 3
    # ]

    data1 = [[
        'B-S', 'I-S', 'I-S',  # ['I', 'love', 'you'] is 'S'
        'O', 'O',  # ['I', 'love'] is not annotated
        'B-NP', 'I-NP',  # ['love', 'you'] is 'VP'
        'B-NP',  # ['I'] is 'NP'
        'B-V',  # ['love'] is 'V'
        'B-NP'  # ['you'] is 'NP'
    ], [
        'B-S', 'I-S', 'I-S',  # ['I', 'love', 'you'] is 'S'
        'O', 'O',  # ['I', 'love'] is not annotated
        'B-VP', 'I-VP',  # ['love', 'you'] is 'VP'
        'B-NP',  # ['I'] is 'NP'
        'B-V',  # ['love'] is 'V'
        'B-V'  # ['you'] is 'NP'
    ]]

    data2 = [
        [
            'B-S', 'I-S', 'I-S',  # S
            'B-NP', 'O', 'O',  # NP
            'O', 'B-VP', 'I-VP',  # VP
            'B-N', 'O', 'O',  # N
            'O', 'B-V', 'B-V',  # V
            # ['I',    'love', 'you' ]
        ],
        [
            'B-S', 'I-S', 'I-S',  # S
            'B-NP', 'O', 'B-NP',  # NP
            'O', 'B-VP', 'I-VP',  # VP
            'B-N', 'O', 'O',  # N
            'O', 'B-V', 'O',  # V
            # ['I',    'love', 'you' ]
        ]
    ]

    print(fleiss_kappa(data1))  # 0.3999999999999999
    print(fleiss_kappa(data2))  # 0.3999999999999999
