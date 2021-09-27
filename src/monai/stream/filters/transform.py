import ctypes
import logging
from typing import Callable
from uuid import uuid4

import cupy
import pyds
from gi.repository import Gst
from stream.errors import BinCreationError
from stream.interface import StreamFilterComponent
from torch.utils.dlpack import from_dlpack, to_dlpack

logger = logging.getLogger(__name__)


class TransformChainComponent(StreamFilterComponent):

    def __init__(self, transform_chain: Callable, name: str = None) -> None:
        self._user_callback = transform_chain
        if not name:
            name = str(uuid4().hex)
        self._name = name

    def initialize(self):
        ucbt = Gst.ElementFactory.make("queue", self.get_name())
        if not ucbt:
            raise BinCreationError(f"Unable to create {self.__class__.__name__} {self.get_name()}")

        self._ucbt = ucbt
        transform_srcpad = self._ucbt.queue2.get_static_pad("src")
        if not transform_srcpad:
            logger.error(f"Unable to obtain a source pad for element {self.__class__.__name__} {self.get_name()}")
            exit(1)

        transform_srcpad.add_probe(Gst.PadProbeType.BUFFER, self.probe_callback, 0)

    def get_name(self):
        return f"{self._name}-usercallbacktransform"

    def get_gst_element(self):
        return self._ucbt

    def probe_callback(self, pad: Gst.PadProbeType, info: Gst.PadProbeCallback, user_data: object):

        inbuf = info.get_buffer()
        if not inbuf:
            print("Unable to get GstBuffer ")
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(inbuf))
        frame_list = batch_meta.frame_meta_list

        while frame_list is not None:

            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break

            owner = None
            data_type, shape, strides, data_ptr, size = pyds.get_nvds_buf_surface_gpu(hash(inbuf), frame_meta.batch_id)
            logger.debug(f"Type: {data_type}, Shape: {shape}, Strides: {strides}, Size: {size}")
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            unownedmem = cupy.cuda.UnownedMemory(
                ctypes.pythonapi.PyCapsule_GetPointer(data_ptr, None),
                size,
                owner,
            )
            memptr = cupy.cuda.MemoryPointer(unownedmem, 0)
            input_cupy_array = cupy.ndarray(
                shape=shape,
                dtype=data_type,
                memptr=memptr,
                strides=strides,
                order='C',
            )
            input_torch_tensor = from_dlpack(input_cupy_array.toDlpack())

            while frame_meta.frame_user_meta_list is not None:

                try:
                    user_meta = pyds.NvDsUserMeta.cast(frame_meta.frame_user_meta_list.data)
                except StopIteration:
                    break

                if (
                    user_meta.base_meta.meta_type
                    != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    continue

                user_meta_data = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                user_data_tensor_layers = []

                for layer_idx in range(user_meta_data.num_output_layers):

                    layer = pyds.get_nvds_LayerInfo(user_meta_data, layer_idx)
                    udata_unownedmem = cupy.cuda.UnownedMemory(
                        ctypes.pythonapi.PyCapsule_GetPointer(layer.buffer, None),
                        ctypes.sizeof(ctypes.c_float) * VIDEO_WIDTH * VIDEO_HEIGHT,
                        owner
                    )
                    udata_memptr = cupy.cuda.MemoryPointer(udata_unownedmem, 0)
                    udata_memptr_cupy = cupy.ndarray(
                        shape=(VIDEO_HEIGHT, VIDEO_WIDTH),
                        dtype=ctypes.c_float,
                        memptr=udata_memptr
                    )

                    user_data_tensor_layers.append(cupy.fromDlpack(to_dlpack(udata_memptr_cupy)))

            stream = cupy.cuda.stream.Stream()
            stream.use()

            user_output_tensor = self._user_callback(input_torch_tensor, user_data_tensor_layers)

            user_output_cupy = cupy.fromDlpack(to_dlpack(user_output_tensor))

            cupy.copyto(user_output_cupy, input_cupy_array)

            stream.synchronize()
