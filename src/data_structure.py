from dataclasses import dataclass
from functools import cached_property

import Levenshtein as lev


@dataclass
class Worker:
    id: int

    def __hash__(self):
        return hash(f'worker-{self.id}')


@dataclass
class Sentence:
    id: int
    content: list[str]

    def __hash__(self):
        return hash(f'sent-{self.id}')


@dataclass
class Annotation:
    worker: Worker
    sent: Sentence

    # For GEC tasks, this is the corrected sentence split into words.
    # Example: ['还', '喜欢', '看', '电影', '、', '运动', '等等', '。']
    content: list[str]

    edit_num: int
    is_gold: float = False
    score: float = -1

    @cached_property
    def edit_seq(self) -> tuple[tuple[str, int, int], ...]:
        # Example: (('delete', 3, 2), ('insert', 4, 4), ('replace', 5, 7))
        return tuple(lev.editops(self.sent.content, self.content))

    def to_system_format(self) -> str:
        return ' '.join(self.content)

    def to_gold_format(self) -> str:
        lines: list[str] = [f'S {" ".join(self.sent.content)}']
        for edit in self.edit_seq:
            if edit[0] == 'replace':
                line = f'A {edit[1]} {edit[1] + 1}|||{edit[0]}|||{self.content[edit[2]]}|||<REQ>|||<COM>|||0'
            elif edit[0] == 'insert':
                line = f'A {edit[1]} {edit[1]}|||{edit[0]}|||{self.content[edit[2]]}|||<REQ>|||<COM>|||0'
            elif edit[0] == 'delete':
                line = f'A {edit[1]} {edit[1] + 1}|||{edit[0]}|||-NONE-|||<REQ>|||<COM>|||0'
            else:
                raise ValueError('Invalid edit.')
            lines.append(line)
        if not self.edit_seq:
            lines.append('A -1 -1|||noop|||-NONE-|||-NONE-|||-NONE-|||1')
        return '\n'.join(lines)

    def to_cherrant_format(self) -> str:
        lines: list[str] = [
            f'S {" ".join(self.sent.content)}',
        ]

        if self.edit_seq:
            lines.append(f'T0-A0 {" ".join(self.content)}')
            for edit in self.edit_seq:
                if edit[0] == 'replace':
                    line = f'A {edit[1]} {edit[1] + 1}|||S|||{self.content[edit[2]]}|||REQUIRED|||-NONE-|||0'
                elif edit[0] == 'insert':
                    line = f'A {edit[1]} {edit[1]}|||M|||{self.content[edit[2]]}|||REQUIRED|||-NONE-|||0'
                elif edit[0] == 'delete':
                    line = f'A {edit[1]} {edit[1] + 1}|||R|||-NONE-|||REQUIRED|||-NONE-|||0'
                else:
                    raise ValueError('Invalid edit.')
                lines.append(line)
        else:
            lines.append(f'T0 没有错误')
            lines.append('A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0')

        return '\n'.join(lines)

    def __hash__(self):
        return hash(f'anno-{hash(self.worker)}-{hash(self.sent)}-{self.content}-{self.edit_num}-{self.is_gold}-{self.score}')

    def __copy__(self):
        return Annotation(self.worker, self.sent, self.content.copy(), self.edit_num, self.is_gold, self.score)
