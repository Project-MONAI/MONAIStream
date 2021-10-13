from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Tuple, Union

from gi.repository import Gst


class StreamComponent(metaclass=ABCMeta):
    @abstractmethod
    def get_name(self):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `get_name`")

    @abstractmethod
    def initialize(self):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `initialize`")

    @abstractmethod
    def get_gst_element(self) -> Union[Gst.Element, Tuple[Gst.Element]]:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `initialize`")


class StreamSourceComponent(StreamComponent):
    @abstractmethod
    def is_live(self):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `is_live`")


class AggregatedSourcesComponent(StreamSourceComponent):
    @abstractmethod
    def get_num_sources(self):
        pass


class StreamFilterComponent(StreamComponent):

    pass


class InferenceFilterComponent(StreamFilterComponent):
    def get_config(self) -> Any:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `get_config`")

    def set_batch_size(self, batch_size: int):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `set_batch_size`")


class MultiplexerComponent(StreamFilterComponent):
    pass


class StreamSinkComponent(StreamComponent):
    def register_probe(self, callback: Callable[[Any, Any], None]):
        self._probe_callback = callback
