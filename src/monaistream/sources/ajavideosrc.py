import logging
from uuid import uuid4

from gi.repository import Gst

from monaistream.errors import BinCreationError
from monaistream.interface import StreamSourceComponent

logger = logging.getLogger(__name__)


class AJAVideoSource(StreamSourceComponent):
    def __init__(self, mode: str, input_mode: str, is_nvmm: bool, name: str = "") -> None:

        if not name:
            name = str(uuid4().hex)

        self._name = name
        self._mode = mode
        self._input_mode = input_mode
        self._is_nvmm = is_nvmm
        self._is_live = True

    def initialize(self):

        aja_video_src_name = f"{self._name}-ajavideosrc"
        aja_video_src = Gst.ElementFactory.make("ajavideosrc", aja_video_src_name)
        if not aja_video_src:
            raise BinCreationError(f"Unable to create source {self.__class__.__name__} with name {aja_video_src}")

        aja_video_src.set_property("mode", self._mode)
        aja_video_src.set_property("input-mode", self._input_mode)
        aja_video_src.set_property("nvmm", self._is_nvmm)

        self._aja_video_src = aja_video_src

    def is_live(self):
        return self._is_live

    def get_name(self):
        return f"{self._name}-ajasource"

    def get_gst_element(self):
        return self._aja_video_src
