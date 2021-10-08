from uuid import uuid4

from gi.repository import Gst
from stream.errors import BinCreationError
from stream.interface import MultiplexerComponent


class NVStreamMux(MultiplexerComponent):
    def __init__(
        self, num_sources: int, width: int, height: int, batched_push_timeout: int = None, name: str = None
    ) -> None:
        if not name:
            name = str(uuid4().hex)

        self._name = name
        self._num_sources = num_sources
        self._width = width
        self._height = height
        self._batched_push_timeout = batched_push_timeout

    def initialize(self):
        streammux = Gst.ElementFactory.make("nvstreammux", self.get_name())
        if not streammux:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._streammux = streammux
        self._streammux.set_property("batch-size", self._num_sources)
        self._streammux.set_property("width", self._width)
        self._streammux.set_property("height", self._height)
        if self._batched_push_timeout:
            self._streammux.set_property("batched-push-timeout", self._batched_push_timeout)

    def get_name(self):
        return f"{self._name}-nvvideoconvert"

    def get_gst_element(self):
        return self._streammux

    def set_is_live(self, is_live: bool):
        self._streammux.set_property("live-source", is_live)
