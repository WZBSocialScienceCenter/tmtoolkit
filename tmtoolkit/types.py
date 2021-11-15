from enum import IntEnum
from typing import Union, Tuple, List, Set


Proportion = IntEnum('Proportion', 'NO YES LOG', start=0)

OrdCollection = Union[tuple, list]
UnordCollection = Union[set, OrdCollection]

OrdStrCollection = Union[Tuple[str], List[str]]
UnordStrCollection = Union[Set[str], OrdStrCollection]

StrOrInt = Union[str, int]
