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

from typing import Optional
from uuid import uuid4

from gi.repository import Gst

from monaistream.errors import BinCreationError
from monaistream.filters.convert import FilterProperties
from monaistream.interface import StreamSourceComponent


class FakeSource(StreamSourceComponent):
    """
    Fake sink component used to terminate a MONAI Stream pipeline.
    """

    def __init__(
            self, name: str = "", num_buffers: int = 1, format_description: Optional[FilterProperties] = None) -> None:
        """
        :param name: the name to assign to this component
        """
        if not name:
            name = str(uuid4().hex)
        self._name = name
        self._num_buffers = num_buffers
        self._format_description = format_description
        self._filter = None

    def initialize(self):
        """
        Initialize the `fakesink` GStreamer element wrapped by this component
        """
        fakesink = Gst.ElementFactory.make("fakesrc", self.get_name())
        if not fakesink:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._fakesource = fakesink
        self._fakesource.set_property("num-buffers", self._num_buffers)

        if self._format_description:
            caps = Gst.Caps.from_string(self._format_description.to_str())
            filter = Gst.ElementFactory.make("capsfilter", f"{self._name}-filter")
            if not filter:
                raise BinCreationError(f"Unable to get the caps for {self.__class__._name} {self.get_name()}")

            filter.set_property("caps", caps)

            self._filter = filter

    def get_gst_element(self):
        """
        Return the raw GStreamer `fakesource` element

        :return: `fakesource` `Gst.Element`
        """
        if self._filter:
            return (self._fakesource, self._filter)
        return self._fakesource

    def get_name(self):
        """
        Get the assigned name of the component

        :return: the name of the component as `str`
        """
        return f"{self._name}-fakevideosource"

    def is_live(self) -> bool:
        return False
