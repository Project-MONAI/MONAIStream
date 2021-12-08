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
from typing import List, Optional, Union
from uuid import uuid4

from gi.repository import Gst

from monaistream.errors import BinCreationError
from monaistream.interface import AggregatedSourcesComponent, StreamSourceComponent

logger = logging.getLogger(__name__)


def _new_pad_handler(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                raise BinCreationError("Unable to get the source bin ghost pad")
        else:
            logger.error("Decodebin did not pick nvidia decoder plugin.")


def _child_added_handler(child_proxy, obj, name, user_data):
    if name.find("decodebin") != -1:
        obj.connect("child-added", _child_added_handler, user_data)
    elif name.find("nvv4l2decoder") != -1:
        obj.set_property("num-extra-surfaces", 4)
        obj.set_property("cudadec-memtype", 0)


class NVAggregatedSourcesBin(AggregatedSourcesComponent):
    """
    An aggregating source bin which, when provided with multiple sources, will batch the inputs from all sources provided
    and send the batched data to downstream components
    """

    def __init__(
        self,
        sources: Union[StreamSourceComponent, List[StreamSourceComponent]],
        output_width: int,
        output_height: int,
        batched_push_timeout: Optional[int] = None,
        name: str = "",
    ) -> None:
        """
        :param sources: One source or a list of sources that are "aggregated" by concatenating the output of all sources in the batch dimension
        :param output_width: The width of the batched output
        :param output_height: The height of the batched output
        :param batched_push_timeout: The timeout in milliseconds to wait for the batch to be formed
        :param name: the desired name of the aggregator component
        """
        if not name:
            name = str(uuid4().hex)

        self._name = name
        self._sources: List[StreamSourceComponent] = sources if isinstance(sources, list) else [sources]
        self._width = output_width
        self._height = output_height
        self._batched_push_timeout = batched_push_timeout
        # if any of the sources are live then so is the wrapper bin
        self._is_live = any([source.is_live() for source in self._sources])

    def initialize(self):
        """
        Initializer method for all provided source components and the `nvstreammux` component which is used
        to batch the output data from all provided sources
        """

        # create the source bin with all the sources specified
        gst_bin = Gst.Bin.new(self.get_name())
        if not gst_bin:
            raise BinCreationError(
                f"Unable to create generic source bin {self.__class__.__name__} with name {self.get_name()}"
            )

        for source in self._sources:
            source.initialize()
            try:
                source.get_gst_element()[-1].connect("pad-added", _new_pad_handler, gst_bin)
            except Exception as e:
                logger.warning(str(e))

            try:
                source.get_gst_element()[-1].connect("child-added", _child_added_handler, gst_bin)
            except Exception as e:
                logger.warning(str(e))

            Gst.Bin.add(gst_bin, source.get_gst_element()[-1])
            bin_pad = gst_bin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
            if not bin_pad:
                raise BinCreationError(
                    f"Unable to add ghost source pad to generic source bin {self.get_name()} "
                    f"{self.__class__.__name__} for source {source.get_name()}"
                )

        self._gst_bin = gst_bin

        # create the stream multiplexer to aggregate all input sources into a batch dimension
        streammux = Gst.ElementFactory.make("nvstreammux", f"{self._name}-nvstreammux")
        if not streammux:
            raise BinCreationError(
                f"Unable to create multiplexer for {self.__class__._name} with name {self.get_name()}"
            )

        self._streammux = streammux
        self._streammux.set_property("batch-size", len(self._sources))
        self._streammux.set_property("width", self._width)
        self._streammux.set_property("height", self._height)
        self._streammux.set_property("live-source", self._is_live)
        if self._batched_push_timeout:
            self._streammux.set_property("batched-push-timeout", self._batched_push_timeout)

        # the bin and muxer will be linked in the composer as they first need to be added to the pipeline

    def is_live(self):
        """
        Returns whether any of the aggregated sources is "live" (e.g. capture card, `rtsp://`, etc.)

        :return: `true` if any of the sources is live
        """
        return self._is_live

    def get_name(self):
        """
        Get the name of the component

        :return: the name as a `str`
        """
        return f"{self._name}-source"

    def get_gst_element(self):
        """
        Return a tuple of GStreamer elements initialized in the components

        :return: a tuple of `Gst.Element`s of types `(gst-bin, nvstreammux)`
        """
        return (self._gst_bin, self._streammux)

    def get_num_sources(self):
        """
        Return the number sources added to this component

        :return: the number of sources assigned to the aggregated component
        """
        return len(self._sources)
