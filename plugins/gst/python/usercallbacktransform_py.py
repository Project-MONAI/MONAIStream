import ctypes
import logging

import cupy
import gi
import pyds
import torch

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstAudio', '1.0')

from gi.repository import GObject, Gst, GstBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NVMM_FORMAT = 'video/x-raw(memory:NVMM),width=[320,7680],height=[240,4320]'
IN_CAPS = Gst.Caps.from_string(NVMM_FORMAT)
OUT_CAPS = Gst.Caps.from_string(NVMM_FORMAT)


def get_buffer_as_tensor(inbuf_info: Gst.MapInfo, batch_id) -> torch.Tensor:

    owner = None

    data_type, shape, strides, data_ptr, size = pyds.get_nvds_buf_surface_gpu(hash(inbuf_info), batch_id, )
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
    unownedmem = cupy.cuda.UnownedMemory(
        ctypes.pythonapi.PyCapsule_GetPointer(data_ptr, None),
        size,
        owner,
    )

    torch_tensor = torch.utils.dlpack.from_dlpack(cupy_array.toDlpack())

    return torch_tensor


def copy_tensor_to_buffer(in_tensor: torch.Tensor, buf: Gst.MapInfo):
    pass


class UserCallbackTransform(GstBase.BaseTransform):

    __gstmetadata__ = (
        'UserCallbackTransform',
        'Filter',
        'MONAI User-defined callback function to use as transform',
        'Alvin Ihsani <aihsani and nvidia dot com>'
    )

    __gproperties__ = {
        "usercallback": (
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
        self._usercallback = None

    def do_get_property(self, prop):
        if prop.name == "usercallback":
            return self._usercallback
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "usercallback":
            self._usercallback = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_transform_caps(self, direction: Gst.PadDirection, caps: Gst.Caps, filter: Gst.Caps) -> Gst.Caps:
        xfm_caps = IN_CAPS if direction == Gst.PadDirection.SRC else OUT_CAPS

        if filter:
            xfm_caps = xfm_caps.intersect(filter)

        return xfm_caps

    def do_set_caps(self, incaps: Gst.Caps, outcaps: Gst.Caps) -> bool:

        in_width, in_height = [incaps.get_structure(0).get_value(v) for v in ['width', 'height']]
        out_width, out_height = [outcaps.get_structure(0).get_value(v) for v in ['width', 'height']]

        if in_height == out_height and in_width == out_width:
            self.set_passthrough(True)

        return True

    def do_transform(self, inbuf: Gst.Buffer, outbuf: Gst.Buffer) -> Gst.FlowReturn:
        if self._usercallback:
            try:
                # convert the incoming frames to tensors and process through user callback
                inbuf_info = inbuf.map(Gst.MapFlags.READ)
                outbuf_info = outbuf.map(Gst.MapFlags.READ)

                batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(inbuf_info))
                frame_list = batch_meta.frame_meta_list

                while frame_list is not None:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                    except StopIteration:
                        break

                    frame_number = frame_meta.frame_num
                    logger.info(f"Processing frame: {frame_number}")

                    inbuf_torch = get_buffer_as_tensor(inbuf_info, frame_meta.batch_id)
                    user_output = self._user_callback(inbuf_torch)
                    copy_tensor_to_buffer(user_output, outbuf_info)

                    try:
                        frame_list = frame_list.next
                    except StopIteration:
                        break

                return Gst.FlowReturn.OK
            except Gst.MapError as e:
                Gst.error("Mapping error: %s" % e)
                return Gst.FlowReturn.ERROR
        else:
            logger.info("No user callback specified")
            return Gst.FlowReturn.OK


GObject.type_register(UserCallbackTransform)
__gstelementfactory__ = ("usercallbacktransform", Gst.Rank.NONE, UserCallbackTransform)
