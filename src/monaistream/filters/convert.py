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

import logging
from typing import Optional, Tuple
from uuid import uuid4

from gi.repository import Gst
from pydantic import BaseModel
from pydantic.types import ConstrainedInt
from typing_extensions import Literal

from monaistream.errors import BinCreationError
from monaistream.interface import StreamFilterComponent

logger = logging.getLogger(__name__)


class SizeConstraint(ConstrainedInt):
    ge = 2
    le = 15360


class ChannelConstraint(ConstrainedInt):
    ge = 1
    ls = 1023


class ConstrainedFramerate(ConstrainedInt):
    ge = 1
    ls = 65535


class FilterProperties(BaseModel):
    memory: Literal["(memory:NVMM)", "-yuv", "(ANY)", ""] = "(memory:NVMM)"
    format: Literal["RGBA", "ARGB", "RGB", "BGR"] = "RGBA"
    width: Optional[SizeConstraint]
    height: Optional[SizeConstraint]
    channels: Optional[ChannelConstraint]
    framerate: Optional[Tuple[ConstrainedFramerate, ConstrainedFramerate]]

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
            format_str = f"{format_str},framerate=(fraction){self.framerate[0]}/{self.framerate[1]}"

        return format_str


class NVVideoConvert(StreamFilterComponent):
    """
    Video converter component for NVIDIA GPU-based video stream
    """

    def __init__(self, format_description: Optional[FilterProperties] = None, name: str = "") -> None:
        """
        Create an :class:`.NVVIdeoConvert` object based on the :class:`.FilterProperties`

        :param filter: the filter property for the video converter component
        :param name: the name of the component
        """
        if not name:
            name = str(uuid4().hex)

        self._name = name
        self._format_description = format_description
        self._filter = None

    def initialize(self):
        """
        Initialize the `nvvideoconvert` GStreamer component
        """
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", self.get_name())
        if not nvvidconv:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._nvvidconv = nvvidconv

        if self._format_description:
            caps = Gst.Caps.from_string(self._format_description.to_str())
            filter = Gst.ElementFactory.make("capsfilter", f"{self._name}-filter")
            if not filter:
                raise BinCreationError(f"Unable to get the caps for {self.__class__._name} {self.get_name()}")

            filter.set_property("caps", caps)

            self._filter = filter

    def get_name(self):
        """
        Get the name of the component

        :return: the name of the component
        """
        return f"{self._name}-nvvideoconvert"

    def get_gst_element(self):
        """
        Get the GStreamer elements initialized with this component

        :return: get a tuple of GStreamer elements of types `(nvvideoconvert, capsfilter)`
        """
        if self._filter:
            return (self._nvvidconv, self._filter)
        return (self._nvvidconv,)
