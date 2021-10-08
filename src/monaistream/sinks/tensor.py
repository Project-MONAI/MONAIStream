import ctypes
from functools import reduce
from typing import Any, List
from uuid import uuid4

import cupy
import pyds
from cupy.core.dlpack import fromDlpack, toDlpack
from gi.repository import Gst
from stream.errors import BinCreationError, StreamProbeRuntimeError
from stream.interface import StreamSinkComponent
from torch.utils.dlpack import from_dlpack, to_dlpack


class NVInferenceMetaToTensor(StreamSinkComponent):
    # TODO: can this be made more configurable so we can get the data from any previous StreamComponent?

    def __init__(
        self, shape: List[int], element_type: Any = ctypes.c_float, synchronize: bool = True, name: str = None
    ) -> None:
        if not name:
            name = str(uuid4().hex)
        self._name = name
        self._element_type = element_type
        self._shape = shape
        self._sync = synchronize

    def initialize(self):
        self._tensor_sink = Gst.ElementFactory.make("fakesink", self.get_name())
        if not self._tensor_sink:
            raise BinCreationError(f"Unable to create sink for {self.__class__.__name__} {self.get_name()}")

        tensor_srcpad = self._tensor_sink.get_static_pad("src")
        if not tensor_srcpad:
            raise BinCreationError(f"Unable to get source pad for {self.__class__.__name__} {self.get_name()}")

        tensor_srcpad.add_probe(Gst.PadProbeType.BUFFER, self._to_tensor, 0)
        self._tensor_sink.set_property("sync", 1 if self._sync else 0)

    def get_gst_element(self):
        return self._tensor_sink

    def get_name(self):
        return f"{self._name}-totensor"

    def _to_tensor(self, pad, info, custom_data):

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            raise StreamProbeRuntimeError("Unable to get buffer for {self.__class__.__name__} {self.get_name()}")

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        frame_meta_list = batch_meta.frame_meta_list

        while frame_meta_list is not None:
            # get frame metadata
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
            except StopIteration:
                break

            user_meta_list = frame_meta.frame_user_meta_list

            while user_meta_list is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    continue

                # get the outputs of the inference
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                # TODO: find out how to get all available outputs
                output_layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

                # convert to CuPy
                owner = None
                data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(
                    hash(gst_buffer), frame_meta.batch_id
                )
                ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                unownedmem = cupy.cuda.UnownedMemory(ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None), size, owner)
                memptr = cupy.cuda.MemoryPointer(unownedmem, 0)
                original_frame = cupy.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order="C")
                result_unownedmem = cupy.cuda.UnownedMemory(
                    ctypes.pythonapi.PyCapsule_GetPointer(output_layer.buffer, None),
                    ctypes.sizeof(self._element_type) * reduce(lambda x, y: x * y, self._shape),
                    owner,
                )
                result_memptr = cupy.cuda.MemoryPointer(result_unownedmem, 0)
                result_cupy = cupy.ndarray(shape=tuple(self._shape), dtype=ctypes.c_float, memptr=result_memptr)
                result_tensor = from_dlpack(toDlpack(result_cupy))

                stream = cupy.cuda.stream.Stream()
                stream.use()

                results_tensor = self._probe_callback(original_frame, result_tensor)
                results_cupy = fromDlpack(to_dlpack(results_tensor))

                original_frame = cupy.ndarray(
                    shape=results_tensor.size(), dtype=data_type, memptr=results_cupy.data, strides=strides, order="C"
                )

                stream.synchronize()

                try:
                    user_meta_list = user_meta_list.next
                except StopIteration:
                    break
            # get next frame
            try:
                frame_meta_list = frame_meta_list.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK
