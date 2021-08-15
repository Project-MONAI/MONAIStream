from uuid import uuid4

from gi.repository import Gst
from stream.interface import StreamSinkComponent
from stream.errors import BinCreationError


class FakeSinkComponent(StreamSinkComponent):

    def __init__(self, name: str = None) -> None:
        if not name:
            name = str(uuid4().hex)
        self._name = name

    def initialize(self):
        fakesink = Gst.ElementFactory.make("fakesink", self.get_name())
        if not fakesink:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._fakesink = fakesink

    def get_gst_element(self):
        return self._fakesink

    def get_name(self):
        return f"{self._name}-fakevideosink"
