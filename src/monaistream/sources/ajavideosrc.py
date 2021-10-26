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
from typing import Optional
from uuid import uuid4

from gi.repository import Gst

from monaistream.errors import BinCreationError
from monaistream.interface import StreamSourceComponent

logger = logging.getLogger(__name__)


class AJAVideoSource(StreamSourceComponent):
    def __init__(
        self,
        mode: str,
        input_mode: str,
        is_nvmm: bool,
        output_width: int,
        output_height: int,
        batched_push_timeout: Optional[int] = None,
        name: str = "",
    ) -> None:

        if not name:
            name = str(uuid4().hex)

        self._name = name
        self._mode = mode
        self._input_mode = input_mode
        self._is_nvmm = is_nvmm
        self._is_live = True
        self._output_width = output_width
        self._output_height = output_height
        self._batched_push_timeout = batched_push_timeout

    def initialize(self):

        aja_video_src_name = f"{self._name}-ajavideosrc"
        aja_video_src = Gst.ElementFactory.make("ajavideosrc", aja_video_src_name)
        if not aja_video_src:
            raise BinCreationError(f"Unable to create source {self.__class__.__name__} with name {aja_video_src}")

        aja_video_src.set_property("mode", self._mode)
        aja_video_src.set_property("input-mode", self._input_mode)
        aja_video_src.set_property("nvmm", self._is_nvmm)

        self._aja_video_src = aja_video_src

        # create the stream multiplexer to aggregate all input sources into a batch dimension
        streammux = Gst.ElementFactory.make("nvstreammux", f"{self._name}-nvstreammux")
        if not streammux:
            raise BinCreationError(
                f"Unable to create multiplexer for {self.__class__._name} with name {self.get_name()}"
            )

        self._streammux = streammux
        self._streammux.set_property("batch-size", 1)
        self._streammux.set_property("width", self._output_width)
        self._streammux.set_property("height", self._output_height)
        self._streammux.set_property("live-source", self._is_live)
        if self._batched_push_timeout:
            self._streammux.set_property("batched-push-timeout", self._batched_push_timeout)

    def is_live(self):
        return self._is_live

    def get_name(self):
        return f"{self._name}-ajasource"

    def get_gst_element(self):
        return (self._aja_video_src, self._streammux)
