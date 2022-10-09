NpSizeAlias = tuple[int, int]
CvSizeAlias = tuple[int, int]


def x(i: NpSizeAlias) -> CvSizeAlias:
    return i


s = (2, 6)
print(x(s))

# ----------

from typing import NewType

NpSizeNT = NewType("NpSizeNT", tuple[int, int])
CvSizeNT = NewType("CvSizeNT", tuple[int, int])


def x2(i: NpSizeNT) -> CvSizeNT:
    return i


s = (2, 6)
print(x2(s))


def x3(i: NpSizeNT) -> CvSizeNT:
    return CvSizeNT(i)


s = NpSizeNT((2, 6))
print(x3(s))
