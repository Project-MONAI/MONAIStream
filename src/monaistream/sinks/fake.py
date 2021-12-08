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

from uuid import uuid4

from gi.repository import Gst

from monaistream.errors import BinCreationError
from monaistream.interface import StreamSinkComponent


class FakeSink(StreamSinkComponent):
    """
    Fake sink component used to terminate a MONAI Stream pipeline.
    """

    def __init__(self, name: str = "") -> None:
        """
        :param name: the name to assign to this component
        """
        if not name:
            name = str(uuid4().hex)
        self._name = name

    def initialize(self):
        """
        Initialize the `fakesink` GStreamer element wrapped by this component
        """
        fakesink = Gst.ElementFactory.make("fakesink", self.get_name())
        if not fakesink:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._fakesink = fakesink

    def get_gst_element(self):
        """
        Return the raw GStreamer `fakesink` element

        :return: `fakesink` `Gst.Element`
        """
        return (self._fakesink,)

    def get_name(self):
        """
        Get the assigned name of the component

        :return: the name of the component as `str`
        """
        return f"{self._name}-fakevideosink"
