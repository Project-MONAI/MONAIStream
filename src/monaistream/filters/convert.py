import logging
from typing import Literal, Optional
from uuid import uuid4

from gi.repository import Gst
from pydantic import BaseModel, conint
from monaistream.errors import BinCreationError
from monaistream.interface import StreamFilterComponent

logger = logging.getLogger(__name__)


class FilterProperties(BaseModel):
    memory: Literal["(memory:NVMM)", "-yuv", "(ANY)"] = "(memory:NVMM)"
    format: Optional[Literal["RGBA", "ARGB", "RGB", "BGR"]] = "RGBA"
    width: Optional[conint(ge=2, le=15360)]
    height: Optional[conint(ge=2, le=15360)]
    channels: Optional[conint(ge=1, le=1023)]
    framerate: Optional[conint(ge=1, le=65535)]

    def to_str(self) -> str:
        format_str = f"video/x-raw{self.memory}"

        if self.format:
            format_str = f"{format_str},format={self.format}"

        if self.width:
            format_str = f"{format_str},width={self.width}"

        if self.height:
            format_str = f"{format_str},height={self.height}"

        if self.channels:
            format_str = f"{format_str},channels={self.channels}"

        if self.framerate:
            format_str = f"{format_str},framerate={self.framerate}"

        return format_str


class NVVideoConvert(StreamFilterComponent):
    def __init__(self, filter: FilterProperties, name: str = None) -> None:
        if not name:
            name = str(uuid4().hex)

        self._name = name
        self._filter = filter

    def initialize(self):
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", self.get_name())
        if not nvvidconv:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._nvvidconv = nvvidconv

        caps = Gst.Caps.from_string(self._filter.to_str())
        filter = Gst.ElementFactory.make("capsfilter", f"{self._name}-filter")
        if not filter:
            raise BinCreationError(f"Unable to get the caps for {self.__class__._name} {self.get_name()}")

        filter.set_property("caps", caps)

        self._filter = filter

    def get_name(self):
        return f"{self._name}-nvvideoconvert"

    def get_gst_element(self):
        return (self._nvvidconv, self._filter)
