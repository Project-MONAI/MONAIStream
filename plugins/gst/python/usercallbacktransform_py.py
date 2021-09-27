import ctypes
import logging

import cupy
import gi
import pyds
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstAudio', '1.0')

from gi.repository import GObject, Gst, GstBase

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# `do_transform` seems to not be used if RGBA convereter is linked
# (NV12 format directly from uridecodebin triggers `do_transform`)
NVMM_FORMAT = 'video/x-raw(memory:NVMM),format=RGBA,width=[320,7680],height=[240,4320]'
IN_CAPS = Gst.Caps.from_string(NVMM_FORMAT)
OUT_CAPS = Gst.Caps.from_string(NVMM_FORMAT)


def get_buffer_as_cupy_matrix(inbuf: Gst.Buffer, batch_id) -> cupy.ndarray:
    owner = None

    data_type, shape, strides, data_ptr, size = pyds.get_nvds_buf_surface_gpu(hash(inbuf), batch_id, )
    logger.debug(f"Type: {data_type}, Shape: {shape}, Strides: {strides}, Size: {size}")
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    unownedmem = cupy.cuda.UnownedMemory(
        ctypes.pythonapi.PyCapsule_GetPointer(data_ptr, None),
        size,
        owner,
    )
    memptr = cupy.cuda.MemoryPointer(unownedmem, 0)
    cupy_array = cupy.ndarray(
        shape=shape,
        dtype=data_type,
        memptr=memptr,
        strides=strides,
        order='C',
    )
    return cupy_array


def get_buffer_as_tensor(inbuf: Gst.Buffer, batch_id: int) -> torch.Tensor:

    cupy_array = get_buffer_as_cupy_matrix(inbuf, batch_id)
    torch_tensor = from_dlpack(cupy_array.toDlpack())

    return torch_tensor


def copy_tensor_to_buffer(in_tensor: torch.Tensor, outbuf: Gst.Buffer, batch_id: int):
    # TODO: fix this thorws SIGSEGV
    outbuf_array = get_buffer_as_cupy_matrix(outbuf, batch_id)

    # TODO: take care of resizing to output buffer shape or throwing a validation error
    outbuf_array[:] = cupy.fromDlpack(to_dlpack(in_tensor))


class UserCallbackTransform(GstBase.BaseTransform):

    __gstmetadata__ = (
        'UserCallbackTransform',
        'Transform',
        'MONAI User-defined callback function to use as transform. Usage: gst-launch-1.0 uridecodebin uri=file:///app/videos/d1_im.mp4 ! mux.sink_0 nvstreammux name=mux width=1260 height=1024 batch-size=1 ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! usercallbacktransform ! nveglglessink sync=True',
        'Alvin Ihsani <aihsani at nvidia dot com>'
    )

    __gproperties__ = {
        "callback": (
            GObject.TYPE_PYOBJECT,  # a `Callable` to be specific
            "User Callback Function",
            "User-specificed Callback Function",
            GObject.ParamFlags.READWRITE,
        ),
    }

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            OUT_CAPS),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            IN_CAPS))

    def __init__(self) -> None:
        super(UserCallbackTransform, self).__init__()
        self._callback = None
        self._in_width = 320
        self._in_height = 240
        self._out_width = 320
        self._out_height = 240

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "callback":
            return self._callback
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == "callback":
            logger.info(f"Setting `{prop.name}` to `{value.__class__.__name__}` type")
            self._callback = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_transform_caps(self, direction: Gst.PadDirection, caps: Gst.Caps, filter: Gst.Caps) -> Gst.Caps:
        xfm_caps = IN_CAPS if direction == Gst.PadDirection.SRC else OUT_CAPS

        if filter:
            xfm_caps = xfm_caps.intersect(filter)

        logger.info(f"do_transform_caps: {xfm_caps}")

        return xfm_caps

    def do_set_caps(self, incaps: Gst.Caps, outcaps: Gst.Caps) -> bool:

        self._in_width, self._in_height = [incaps.get_structure(0).get_value(v) for v in ['width', 'height']]
        self._out_width, self._out_height = [outcaps.get_structure(0).get_value(v) for v in ['width', 'height']]

        logger.info(f"do_set_caps: ({self._in_width},{self._in_height}) -> ({self._out_width},{self._out_height})")

        # self.set_passthrough(False)
        # self.set_in_place(False)
        return True

    def do_fixate_caps(self, direction: Gst.PadDirection, caps: Gst.Caps, othercaps: Gst.Caps) -> Gst.Caps:
        """
            caps: initial caps
            othercaps: target caps
        """
        logger.info(f"do_fixate_caps: {caps} -> {othercaps}")
        if direction == Gst.PadDirection.SRC:
            return othercaps.fixate()
        else:
            new_format = othercaps.get_structure(0).copy()

            new_format.fixate_field_nearest_int("width", self._out_width)
            new_format.fixate_field_nearest_int("height", self._out_height)
            new_caps = Gst.Caps.new_empty()
            new_caps.append_structure(new_format)

            return new_caps.fixate()

    def do_transform(self, inbuf: Gst.Buffer, outbuf: Gst.Buffer) -> Gst.FlowReturn:

        logger.info("Invoking: do_transform")

        if self._callback is None:
            logger.info("No user callback specified")
            return Gst.FlowReturn.OK

        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(inbuf))
            frame_list = batch_meta.frame_meta_list

            while frame_list is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                    logger.debug(
                        f"Frame meta:\nFrame #: {frame_meta.frame_num}\nBatch id: {frame_meta.batch_id}\n{frame_meta}")
                except StopIteration:
                    break

                frame_number = frame_meta.frame_num
                logger.info(f"Processing frame: {frame_number}")

                inbuf_torch = get_buffer_as_tensor(inbuf, frame_meta.batch_id)
                logger.debug(
                    f"Converted buffer to tensor: Shape {inbuf_torch.size()}, Type: {inbuf_torch.type()}, Device: {inbuf_torch.device.type}")

                user_output = self._callback(inbuf_torch)
                logger.debug(
                    f"Callback output: Shape {user_output.size()}, Type: {user_output.type()}, Device: {user_output.device.type}")

                copy_tensor_to_buffer(user_output, outbuf, frame_meta.batch_id)
                logger.debug("Copied user output to output buffer")

                try:
                    frame_list = frame_list.next
                except StopIteration:
                    break

            return Gst.FlowReturn.OK
        except Exception as e:
            Gst.error(f"{self.__class__.__name__}::do_transform error: %s" % e)
            raise e
            return Gst.FlowReturn.ERROR


GObject.type_register(UserCallbackTransform)
__gstelementfactory__ = ("usercallbacktransform", Gst.Rank.NONE, UserCallbackTransform)
