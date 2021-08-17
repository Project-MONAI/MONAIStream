import ctypes

import cupy
import gi
import pyds
import torch

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstAudio', '1.0')

from gi.repository import GObject, Gst, GstBase, GLib


DEFAULT_WIDTH = 1264
DEFAULT_HEIGHT = 1080


def DEFAULT_CALLBACK(x):
    return x


FORMATS = "{RGBx,BGRx,xRGB,xBGR,RGBA,BGRA,ARGB,ABGR,RGB,BGR,GRAY8}"
FIXED_CAPS = Gst.Caps.from_string(
    f'video/x-raw,format={FORMATS}')


class UserCallbackTransform(GstBase.BaseTransform):

    __gstmetadata__ = (
        'UserCallbackTransform', 'Transform',
        'gst-python MONAI transform insertion element',
        'aihsani')

    __gproperties__ = {
        "usercallback": (
            GObject.TYPE_PYOBJECT,  # a `Callable` to be specific
            "UserCallback",
            "User-specificed Callback Function",
            DEFAULT_CALLBACK,
            GObject.ParamFlag.READWRITE,
        ),
        "width": (
            GObject.TYPE_INT,
            "Width",
            "Video Width in Pixels",
            0,
            GLib.MAXINT,
            DEFAULT_WIDTH,
            GObject.ParamFlag.READWRITE,
        ),
        "height": (
            GObject.TYPE_INT,
            "Height",
            "Video Height in Pixels",
            0,
            GLib.MAXINT,
            DEFAULT_HEIGHT,
            GObject.ParamFlag.READWRITE,
        ),
    }

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            FIXED_CAPS),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            FIXED_CAPS))

    def __init__(self) -> None:
        GstBase.BaseSrc.__init__(self)

        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.usercallback = DEFAULT_CALLBACK

    def do_get_property(self, prop):
        if prop.name == 'width':
            return self.width
        elif prop.name == "height":
            return self.height
        elif prop.name == "usercallback":
            return self.usercallback
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == 'width':
            self.width = value
        elif prop.name == "height":
            self.height = value
        elif prop.name == "usercallback":
            self.usercallback = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_transform_ip(self, inbuf: Gst.Buffer):
        try:
            with inbuf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:

                batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info))
                frame_list = batch_meta.frame_meta_list

                while frame_list is not None:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                    except StopIteration:
                        break

                    # frame_number = frame_meta.frame_num
                    user_metadata_list = frame_meta.frame_user_meta_list

                    while user_metadata_list is not None:
                        try:
                            user_metadata = pyds.NvDsUserMeta.cast(user_metadata_list.data)
                        except StopIteration:
                            break

                        # TODO: consider generalizing this to perhaps something beyond nvinfer outputs
                        if user_metadata.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                            continue

                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_metadata.user_meta_data)
                        owner = None

                        orig_data_type, orig_shape, orig_strides, orig_ptr, orig_size = pyds.get_nvds_buf_surface_gpu(
                            hash(info),
                            frame_meta.batch_id,
                        )
                        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                        unownedmem = cupy.cuda.UnownedMemory(
                            ctypes.pythonapi.PyCapsule_GetPointer(orig_ptr, None),
                            orig_size,
                            owner,
                        )
                        orig_memptr = cupy.cuda.MemoryPointer(unownedmem, 0)
                        orig_image_cupy = cupy.ndarray(
                            shape=orig_shape,
                            dtype=orig_data_type,
                            memptr=orig_memptr,
                            strides=orig_strides,
                            order='C',
                        )
                        unownedmem = cupy.cuda.UnownedMemory(
                            ctypes.pythonapi.PyCapsule_GetPointer(orig_ptr, None),
                            orig_size,
                            owner,
                        )

                        orig_image_torch = torch.utils.dlpack.from_dlpack(orig_image_cupy.toDlpack())

                        output_torch_list = []
                        for output_idx in range(tensor_meta.num_output_layers):

                            output_meta = pyds.get_nvds_LayerInfo(tensor_meta, output_idx)
                            output_unownedmem = cupy.cuda.UnownedMemory(
                                ctypes.pythonapi.PyCapsule_GetPointer(output_meta.buffer, None),
                                ctypes.sizeof(ctypes.c_float) * VIDEO_WIDTH * VIDEO_HEIGHT, owner)

                            output_memptr = cupy.cuda.MemoryPointer(output_unownedmem, 0)
                            output_cupy = cupy.ndarray(shape=(), dtype=ctypes.c_float, memptr=output_memptr)

                            output_torch = torch.utils.dlpack.from_dlpack(output_cupy.toDlpack())

                            output_torch_list.append(output_torch)

                        stream = cupy.cuda.stream.Stream()
                        stream.use()

                        # TODO: generalize this to return values
                        #       here it is assumed that user data manipulation happens in place
                        self._user_callback(orig_image_torch, *output_torch_list)

                        stream.synchronize()

                        try:
                            user_metadata_list = user_metadata_list.next
                        except StopIteration:
                            break

                    try:
                        frame_list = frame_list.next
                    except StopIteration:
                        break

                return Gst.FlowReturn.OK
        except Gst.MapError as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR


GObject.type_register(UserCallbackTransform)
__gstelementfactory__ = ("usercallbacktransform", Gst.Rank.NONE, UserCallbackTransform)
