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
from typing_extensions import Literal

from monaistream.errors import BinCreationError
from monaistream.interface import StreamSourceComponent


class TestVideoSource(StreamSourceComponent):
    """
    Test source component used to generate data in a MONAI Stream pipeline.
    """

    def __init__(
        self,
        name: str = "",
        num_buffers: int = 1,
        is_live: bool = False,
        pattern: Literal["black", "white", "smpte75"] = "black",
    ) -> None:
        """
        :param name: the name to assign to this component
        """
        if not name:
            name = str(uuid4().hex)
        self._name = name
        self._num_buffers = num_buffers
        self._is_live = is_live
        self._pattern = pattern

    def initialize(self):
        """
        Initialize the `videotestsrc` GStreamer element wrapped by this component
        """
        testvideosrc = Gst.ElementFactory.make("videotestsrc", self.get_name())
        if not testvideosrc:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._testvideosrc = testvideosrc
        self._testvideosrc.set_property("num-buffers", self._num_buffers)
        self._testvideosrc.set_property("pattern", self._pattern)
        self._testvideosrc.set_property("is-live", self._is_live)

    def get_gst_element(self):
        """
        Return the raw GStreamer `testvideosrc` element

        :return: `tesvideosrc` `Gst.Element`
        """
        return (self._testvideosrc,)

    def get_name(self):
        """
        Get the assigned name of the component

        :return: the name of the component as `str`
        """
        return f"{self._name}-testvideosource"

    def is_live(self) -> bool:
        return False
