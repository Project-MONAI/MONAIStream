from uuid import uuid4

from gi.repository import Gst
from stream.errors import BinCreationError
from stream.interface import StreamFilterComponent


class NVRGBAFilter(StreamFilterComponent):

    def __init__(self, name: str = None) -> None:
        if not name:
            name = str(uuid4().hex)

        self._name = name

    def initialize(self):

        rgba_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        filter = Gst.ElementFactory.make("capsfilter", f"{self._name}-filter")
        if not filter:
            raise BinCreationError(f"Unable to get the caps for {self.__class__._name} {self.get_name()}")

        filter.set_property("caps", rgba_caps)

        self._filter = filter

    def get_name(self):
        return f"{self._name}-rgbafilter"

    def get_gst_element(self):
        return self._filter


class NVVideoConvert(StreamFilterComponent):

    def __init__(self, name: str = None) -> None:
        if not name:
            name = str(uuid4().hex)

        self._name = name

    def initialize(self):
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", self.get_name())
        if not nvvidconv:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._nvvidconv = nvvidconv

    def get_name(self):
        return f"{self._name}-nvvideoconvert"

    def get_gst_element(self):
        return self._nvvidconv
