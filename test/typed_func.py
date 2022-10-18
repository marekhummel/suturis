from typing import TypeVar


T = TypeVar("T")


def make_pair(exp_type: type[T]) -> T:
    if exp_type == int:
        return 1
    elif exp_type == str:
        return "4"


print(make_pair(int))
print(make_pair(str))
