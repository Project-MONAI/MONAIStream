import logging
from typing import List
from uuid import uuid4

from gi.repository import Gst
from stream.errors import BinCreationError
from stream.interface import AggregatedSourcesComponent, StreamSourceComponent

logger = logging.getLogger(__name__)


def _new_pad_handler(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
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
        obj.set_property('cudadec-memtype', 0)


class NVAggregatedSourcesBin(AggregatedSourcesComponent):

    def __init__(
        self,
        sources: List[StreamSourceComponent],
        name: str = None
    ) -> None:

        if not name:
            name = str(uuid4().hex)

        self._name = name
        self._sources = sources
        # if any of the sources are live then so is the wrapper bin
        self._is_live = any([source.is_live() for source in sources])

    def initialize(self):
        gst_bin = Gst.Bin.new(self.get_name())
        if not gst_bin:
            raise BinCreationError(
                f"Unable to create generic source bin {self.__class__.__name__} "
                f"with name {self.get_name()}"
            )

        for source in self._sources:
            source.initialize()
            source.get_gst_element().connect("pad-added", _new_pad_handler, gst_bin)
            source.get_gst_element().connect("child-added", _child_added_handler, gst_bin)

            Gst.Bin.add(gst_bin, source.get_gst_element())
            bin_pad = gst_bin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
            if not bin_pad:
                raise BinCreationError(
                    f"Unable to add ghost source pad to generic source bin {self.get_name()} "
                    f"{self.__class__.__name__} for source {source.get_name()}"
                )

        self._gst_bin = gst_bin

    def is_live(self):
        return self._is_live

    def get_name(self):
        return f"{self._name}-source"

    def get_gst_element(self):
        return self._gst_bin

    def get_num_sources(self):
        return len(self._sources)
