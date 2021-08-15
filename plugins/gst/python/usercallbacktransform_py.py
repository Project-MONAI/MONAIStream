import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstAudio', '1.0')

from gi.repository import GObject, Gst, GstBase

FORMATS = "{RGBx,BGRx,xRGB,xBGR,RGBA,BGRA,ARGB,ABGR,RGB,BGR}"
FIXED_CAPS = Gst.Caps.from_string(
    f'video/x-raw,format={FORMATS}')


class UserCallbackTransform(GstBase.BaseTransform):

    __gstmetadata__ = (
        'UserCallbackTransform', 'Transform',
        'gst-python MONAI transform insertion element',
        'aihsani')

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            FIXED_CAPS),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            FIXED_CAPS))

    def do_transform_ip(self, inbuf: Gst.Buffer):
        try:
            with inbuf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:

                A = np.ndarray(shape = (self.height, self.width), dtype = np.uint8, buffer = info.data)
                A[:] = np.invert(A)

                return Gst.FlowReturn.OK
        except Gst.MapError as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR


GObject.type_register(UserCallbackTransform)
__gstelementfactory__ = ("usercallbacktransform", Gst.Rank.NONE, UserCallbackTransform)
