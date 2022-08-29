from typing import TypeVar, Type

from logging import 


T = TypeVar("T")


def make_pair(exp_type: Type[T]) -> T:
    if exp_type == int:
        return 1
    elif exp_type == str:
        return "4"


print(make_pair(int))
print(make_pair(str))
