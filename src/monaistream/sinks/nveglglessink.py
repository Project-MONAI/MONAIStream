from uuid import uuid4

from gi.repository import Gst
from stream.errors import BinCreationError
from stream.interface import StreamSinkComponent


class NVEglGlesSink(StreamSinkComponent):
    def __init__(self, name: str = None, sync: bool = False) -> None:
        if not name:
            name = str(uuid4().hex)
        self._name = name
        self._sync = sync

    def initialize(self):
        eglsink = Gst.ElementFactory.make("nveglglessink", self.get_name())
        if not eglsink:
            raise BinCreationError(f"Unable to create {self.__class__.__name__} {self.get_name()}")

        self._elgsink = eglsink
        self._elgsink.set_property("sync", 1 if self._sync else 0)

    def get_name(self):
        return f"{self._name}-usercallbacktransform"

    def get_gst_element(self):
        return self._elgsink
