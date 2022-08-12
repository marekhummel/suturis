from enum import Enum
import multiprocessing as mp
import multiprocessing.shared_memory as mpsm
import numpy as np


class Param(Enum):
    TranslatedBase = 1
    WarpedQuery = 2
    StitchingMask = 3
    SeamCorners = 4
    SeamMatrix = 5


class LocalParams:
    def __init__(self, img1, img2, mask, seam_corner, seammat) -> None:
        self.translated_base = img1
        self.warped_query = img2
        self.stitch_mask = mask
        self.seam_corners = seam_corner
        self.seam_matrix = seammat

    # def __init__(self, shared: "SharedParams") -> None:
    #     self.translated_base = self._transfer_np_array(*shared.translated_base)
    #     self.warped_query = self._transfer_np_array(*shared.warped_query)
    #     self.stitch_mask = self._transfer_np_array(*shared.stitch_mask)
    #     self.seam_corners = self._transfer_np_array(*shared.seam_corners)
    #     self.seam_matrix = self._transfer_np_array(*shared.seam_corners)

    def _transfer_np_array(self, mp_name, mp_shape) -> np.ndarray:
        shm_name = mp_name.value
        shape = mp_shape.value
        existing_shm = mpsm.SharedMemory(name=shm_name)
        a = np.ndarray(shape, np.float64, buffer=existing_shm.buf)
        local_a = a[:]
        existing_shm.close()
        existing_shm.unlink()
        return local_a


class SharedParams:
    def __init__(self) -> None:
        self.translated_base = (mp.Array("c", 15), mp.Array("i", 3))
        self.warped_query = (mp.Array("c", 15), mp.Array("i", 3))
        self.stitch_mask = (mp.Array("c", 15), mp.Array("i", 3))
        self.seam_matrix = (mp.Array("c", 15), mp.Array("i", 3))
        self.seam_corners = mp.Array("i", 4)

        self._shared_memory_objs = set()

    def set_value(self, p: "Param", value) -> None:
        if p in [
            Param.TranslatedBase,
            Param.WarpedQuery,
            Param.StitchingMask,
            Param.SeamMatrix,
        ]:
            memory_name = self._transfer_np_array(value)
            self.translated_base[0].value = memory_name
            self.translated_base[1].value = value.shape
        elif p == Param.SeamCorners:
            start, end = value
            self.seam_corners.value = [start[0], start[1], end[0], end[1]]

    def close(self):
        for shm in self._shared_memory_objs:
            shm.close()

    def _transfer_np_array(self, a: np.ndarray) -> str:
        shm = mpsm.SharedMemory(create=True, size=a.nbytes)
        self._shared_memory_objs.add(shm)
        buffed = np.ndarray(a.shape, dtype=np.float64, buffer=shm.buf)
        buffed[:] = a.astype(np.float64)[:]
        return shm.name
