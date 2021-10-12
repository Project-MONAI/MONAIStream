import logging
from uuid import uuid4

from gi.repository import Gst

from monaistream.errors import BinCreationError
from monaistream.interface import StreamSourceComponent

logger = logging.getLogger(__name__)


class URISource(StreamSourceComponent):
    def __init__(self, uri: str, name: str = None) -> None:

        if not name:
            name = str(uuid4().hex)

        self._name = name
        self._uri = uri
        self._is_live = uri.find("rtsp://") == 0

    def initialize(self):

        uri_decode_bin_name = f"{self._name}-uridecodebin"
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", uri_decode_bin_name)
        if not uri_decode_bin:
            raise BinCreationError(
                f"Unable to create source {self.__class__.__name__} with name {uri_decode_bin} for URI {self._uri}"
            )

        uri_decode_bin.set_property("uri", self._uri)

        self._uri_decode_bin = uri_decode_bin

    def is_live(self):
        return self._is_live

    def get_name(self):
        return f"{self._name}-source"

    def get_gst_element(self):
        return self._uri_decode_bin
