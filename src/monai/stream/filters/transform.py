from typing import Callable
from uuid import uuid4

from gi.repository import Gst
from stream.errors import BinCreationError
from stream.interface import StreamFilterComponent


class TransformChainComponent(StreamFilterComponent):

    def __init__(self, transform_chain: Callable, name: str = None) -> None:
        self._user_callback = transform_chain
        if not name:
            name = str(uuid4().hex)
        self._name = name

    def initialize(self):
        ucbt = Gst.ElementFactory.make('usercallbacktransform', self.get_name())
        if not ucbt:
            raise BinCreationError(f"Unable to create {self.__class__.__name__} {self.get_name()}")

        self._ucbt = ucbt
        self._ucbt.set_property("usercallback", self._user_callback)

    def get_name(self):
        return f"{self._name}-usercallbacktransform"

    def get_gst_element(self):
        return self._ucbt
