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


class StreamSinkComponent(StreamComponent):
    def register_probe(self, callback: Callable[[Any, Any], None]):
        self._probe_callback = callback
