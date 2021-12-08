################################################################################
# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import ctypes
import logging
from typing import Callable, Dict
from uuid import uuid4

import cupy
from gi.repository import Gst

import pyds
from monaistream.errors import BinCreationError
from monaistream.interface import StreamFilterComponent
from monaistream.filters.util import get_nvdstype_npsize, get_nvdstype_size


logger = logging.getLogger(__name__)


class TransformChainComponentCupy(StreamFilterComponent):
    """
    The `TransformChainComponentCupy` allows users to plugin a `Callable` into the MONAI Stream pipeline.
    The user-specified callable must receive a Cupy array or list of Cupy arrays, and return one single Cupy array as the result.
    """

    def __init__(self, transform_chain: Callable, output_label: str, name: str = "") -> None:
        """
        :param transform_chain: a `Callable` object such as `monai.transforms.compose.Compose`
        """
        self._user_callback = transform_chain
        if not name:
            name = str(uuid4().hex)
        self._name = name
        self._input_labels = ["ORIGINAL_IMAGE"]
        self._output_label = output_label

    def initialize(self):
        """
        Initializes the GStreamer element wrapped by this component, which is a `queue` element
        """
        ucbt = Gst.ElementFactory.make("queue", self.get_name())
        if not ucbt:
            raise BinCreationError(f"Unable to create {self.__class__.__name__} {self.get_name()}")

        self._ucbt = ucbt
        transform_sinkpad = self._ucbt.get_static_pad("sink")
        if not transform_sinkpad:
            logger.error(f"Unable to obtain a sink pad for element {self.__class__.__name__} {self.get_name()}")
            exit(1)

        transform_sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.probe_callback, 0)

    def get_name(self):
        """
        Get the name assigned to the component

        :return: the name as a `str`
        """
        return f"{self._name}-usercallbacktransform"

    def get_gst_element(self):
        """
        Return the GStreamer element

        :return: the raw `queue` `Gst.Element`
        """
        return (self._ucbt,)

    def probe_callback(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object):
        """
        A wrapper function for the `transform_chain` callable set in the constructor. Performs conversion of GStreamer data
        (a Gst.Buffer in the GPU) to a Cupy array before the user-specified `transform_chain` is called; the result of `transform_chain`
        is converted back to a `Gst.Buffer` and written to the original input buffer. NOTE: The size of the output must be the same
        as or smaller than the input buffer.
        """
        inbuf = info.get_buffer()
        if not inbuf:
            logger.error("Unable to get GstBuffer ")
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
                order="C",
            )

            user_data_cupy_layers = []
            user_meta_list = frame_meta.frame_user_meta_list
            if user_meta_list is not None:

                try:
                    user_meta = pyds.NvDsUserMeta.cast(frame_meta.frame_user_meta_list.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    continue

                user_meta_data = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                logger.debug(f"# output layers: {user_meta_data.num_output_layers}")

                for layer_idx in range(user_meta_data.num_output_layers):

                    layer = pyds.get_nvds_LayerInfo(user_meta_data, layer_idx)

                    layer_dims = []
                    elems = 1
                    for dim in range(layer.dims.numDims):
                        layer_dims.append(layer.dims.d[dim])
                        elems *= layer.dims.d[dim]

                    if not layer.isInput:
                        self._input_labels.append(layer.layerName)

                    udata_unownedmem = cupy.cuda.UnownedMemory(
                        ctypes.pythonapi.PyCapsule_GetPointer(layer.buffer, None),
                        get_nvdstype_size(layer.dataType) * elems,
                        owner,
                    )
                    udata_memptr = cupy.cuda.MemoryPointer(udata_unownedmem, 0)
                    udata_memptr_cupy = cupy.ndarray(
                        shape=tuple(layer_dims),
                        dtype=get_nvdstype_npsize(layer.dataType),
                        memptr=udata_memptr,
                    )

                    logger.debug(
                        f"Layer Name: {layer.layerName}, Is Input: {layer.isInput},"
                        f" Dims: {layer_dims}, Data Type: {layer.dataType}"
                    )

                    user_data_cupy_layers.append(udata_memptr_cupy)

            stream = cupy.cuda.stream.Stream()
            stream.use()

            user_input_data: Dict[str, cupy.ndarray] = {
                label: data for label, data in zip(self._input_labels, [input_cupy_array, *user_data_cupy_layers])
            }

            try:

                user_output_cupy = self._user_callback(user_input_data)[self._output_label]
                cupy.copyto(input_cupy_array, user_output_cupy)

            except Exception as e:
                logger.exception(e)
                return Gst.PadProbeReturn.HANDLED

            stream.synchronize()

            return Gst.PadProbeReturn.OK
