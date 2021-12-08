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


class NVEglGlesSink(StreamSinkComponent):
    """
    NVIDIA video viewport sink to visualize results of MONAI Stream pipeline.
    """

    def __init__(self, name: str = "", sync: bool = False) -> None:
        """
        :param sync: `True` is the frames should synchronize with the source, and `False` otherwise
        :param name: the name to assign to this component
        """
        if not name:
            name = str(uuid4().hex)
        self._name = name
        self._sync = sync

    def initialize(self):
        """
        Initialize the GStreamer `nveglglessink` element wrapped by this component
        """
        eglsink = Gst.ElementFactory.make("nveglglessink", self.get_name())
        if not eglsink:
            raise BinCreationError(f"Unable to create {self.__class__.__name__} {self.get_name()}")

        self._elgsink = eglsink
        self._elgsink.set_property("sync", 1 if self._sync else 0)

    def get_name(self):
        """
        Get the name assigned to this component

        :return: the name of the component as `str`
        """
        return f"{self._name}-usercallbacktransform"

    def get_gst_element(self):
        """
        Return the raw GStreamer `nveglglessink` element

        :return: `nveglglessink` `Gst.Element`
        """
        return (self._elgsink,)
