from uuid import uuid4

from gi.repository import Gst

from monaistream.errors import BinCreationError
from monaistream.interface import StreamSinkComponent


class FakeSink(StreamSinkComponent):
    """
    Fake sink component used to terminate a MONAI Stream pipeline.
    """

    def __init__(self, name: str = "") -> None:
        """
        :param name: the name to assign to this component
        """
        if not name:
            name = str(uuid4().hex)
        self._name = name

    def initialize(self):
        """
        Initialize the `fakesink` GStreamer element wrapped by this component
        """
        fakesink = Gst.ElementFactory.make("fakesink", self.get_name())
        if not fakesink:
            raise BinCreationError(f"Unable to create {self.__class__._name} {self.get_name()}")

        self._fakesink = fakesink

    def get_gst_element(self):
        """
        Return the raw GStreamer `fakesink` element

        :return: `fakesink` `Gst.Element`
        """
        return self._fakesink

    def get_name(self):
        """
        Get the assigned name of the component

        :return: the name of the component as `str`
        """
        return f"{self._name}-fakevideosink"
