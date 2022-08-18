from enum import Enum
import multiprocessing as mp
import multiprocessing.shared_memory as mpsm
import numpy.ctypeslib as npc
import ctypes
import numpy as np


class CParamsStruct(ctypes.Structure):
    _fields_ = [
        ("input_base", npc.ndpointer()),
        ("input_query", npc.ndpointer()),
        ("translated_base", npc.ndpointer()),
        ("warped_query", npc.ndpointer()),
        ("stitching_mask", npc.ndpointer()),
        ("seam_corners", ctypes.c_int * 4),
        ("seam_matrix", npc.ndpointer())
    ]


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

