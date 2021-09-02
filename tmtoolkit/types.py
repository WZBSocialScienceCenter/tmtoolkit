from typing import Union, Tuple, List, Set

OrdCollection = Union[tuple, list]
UnordCollection = Union[set, OrdCollection]

OrdStrCollection = Union[Tuple[str], List[str]]
UnordStrCollection = Union[Set[str], OrdStrCollection]
