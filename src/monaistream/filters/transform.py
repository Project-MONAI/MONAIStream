import ctypes
import logging
from typing import Callable, Dict, List, Union
from uuid import uuid4

import cupy
from gi.repository import Gst
from torch import Tensor
from torch.utils.dlpack import from_dlpack, to_dlpack

import pyds
from monaistream.errors import BinCreationError
from monaistream.interface import StreamFilterComponent

logger = logging.getLogger(__name__)


DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240


class TransformChainComponent(StreamFilterComponent):
    """
    The `TransformChainComponent` allows users to plugin a MONAI transformation pipeline into the MONAI `StreamCompose` component 
    """
    def __init__(
        self,
        transform_chain: Callable,
        input_labels: List[str] = [],
        output_label: str = "",
        name: str = "",
    ) -> None:
        """
        :param transform_chain: a `Callable` object such as `monai.transforms.compose.Compose`
        :param input_labels: the label keys we want to assign to the inputs to this component
        :param output_labels: the label key to select the output from this component
        """
        self._user_callback = transform_chain
        if not name:
            name = str(uuid4().hex)
        self._name = name
        self._input_labels = input_labels
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
        return self._ucbt

    def probe_callback(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: object):
        """
        A wrapper function for the `transform_chain` callable set in the constructor. Performs conversion of GStreamer data
        to a `torch.Tensor` before the user-specified `transform_chain` is called, and converts back from the user's `torch.Tensor`
        result. 
        """

        inbuf = info.get_buffer()
        if not inbuf:
            logger.error("Unable to get GstBuffer ")
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(inbuf))
        frame_list = batch_meta.frame_meta_list

        caps = pad.get_current_caps()
        if not caps:
            caps = pad.get_allowed_caps()

        caps_struct = caps.get_structure(0)
        success, image_width = caps_struct.get_int("width")
        if not success:
            image_width = DEFAULT_WIDTH

        success, image_height = caps_struct.get_int("height")
        if not success:
            image_height = DEFAULT_HEIGHT

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
            input_torch_tensor = from_dlpack(input_cupy_array.toDlpack())

            user_data_tensor_layers = []
            user_meta_list = frame_meta.frame_user_meta_list
            while user_meta_list is not None:

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
                    udata_unownedmem = cupy.cuda.UnownedMemory(
                        ctypes.pythonapi.PyCapsule_GetPointer(layer.buffer, None),
                        ctypes.sizeof(ctypes.c_float) * image_width * image_height,
                        owner,
                    )
                    udata_memptr = cupy.cuda.MemoryPointer(udata_unownedmem, 0)
                    udata_memptr_cupy = cupy.ndarray(
                        shape=(image_height, image_width),
                        dtype=ctypes.c_float,
                        memptr=udata_memptr,
                    )

                    user_data_tensor_layers.append(from_dlpack(udata_memptr_cupy.toDlpack()))
                    # logger.debug(f"Adding user data, size: {user_data_tensor_layers[-1].size()}")
                    # logger.debug(f"Size of user data: {len(user_data_tensor_layers)}")

                break

            stream = cupy.cuda.stream.Stream()
            stream.use()

            user_input_data: Union[List[Tensor], Dict[str, Tensor]] = []

            if self._input_labels:
                user_input_data = {
                    label: data
                    for label, data in zip(self._input_labels, [input_torch_tensor, *user_data_tensor_layers])
                }
            else:
                user_input_data = [input_torch_tensor, *user_data_tensor_layers]

            if not self._output_label:
                user_output_tensor = list(self._user_callback(user_input_data).values())[0]
            else:
                user_output_tensor = self._user_callback(user_input_data)[self._output_label]

            user_output_cupy = cupy.fromDlpack(to_dlpack(user_output_tensor))

            cupy.copyto(input_cupy_array, user_output_cupy)

            stream.synchronize()

            return Gst.PadProbeReturn.OK
