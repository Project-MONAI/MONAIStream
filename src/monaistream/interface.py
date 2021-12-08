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
from typing import Any, Tuple

from gi.repository import Gst


class StreamComponent(metaclass=ABCMeta):
    """
    Default class for all streaming components. All components to added in `StreamCompose` must inherit from `StreamComponent`
    """

    @abstractmethod
    def get_name(self):
        """
        Get the name of the datastore

        :return: the name of the component
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `get_name`")

    @abstractmethod
    def initialize(self):
        """
        Initialize the GStreamer element which this `StreamComponent` wraps
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `initialize`")

    @abstractmethod
    def get_gst_element(self) -> Tuple[Gst.Element]:
        """
        Get GStreamer element or elements initialized in the `initalize` method

        :return: a tuple of `Gst.Element`s or a single `Gst.Element` when only one exists
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `initialize`")


class StreamSourceComponent(StreamComponent):
    """
    Default class for all source components
    """

    @abstractmethod
    def is_live(self) -> bool:
        """
        Determine if the source component is live (e.g. `rtsp://` or capture card)

        :return: whether the source component is a live stream (bool)
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `is_live`")


class AggregatedSourcesComponent(StreamSourceComponent):
    """
    A special component which should be inherited when creating a multi-source component
    (see :class:`monaistream.sources.NVAggregatedSourcesComponent`)
    """

    @abstractmethod
    def get_num_sources(self) -> int:
        """
        Determine the number of source included in this component

        :return: the number of sources aggregated in this component (int)
        """
        pass


class StreamFilterComponent(StreamComponent):
    """
    The interface for filtering components in MONAI Streak SDK. Filter components that transform data, but do not
    generate or consume data without generating new data.
    """

    pass


class InferenceFilterComponent(StreamFilterComponent):
    """
    An inference (filter) component abstracting basic methods for components that perform inference
    (e.g. :class:`monaistream.filters.infer.NVInferServer`).
    """

    def get_config(self) -> Any:
        """
        Get the configuration of the inference component

        :return: An object representing the configuration of the component (if any)
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `get_config`")

    def set_batch_size(self, batch_size: int):
        """
        Set the batch size for the inference

        :param batch_size: the batch size which will be used to perform inference
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement `set_batch_size`")


class StreamSinkComponent(StreamComponent):
    """
    The interface for all sink components in MONAI Stream SDK. Sink component consume data without
    producing any consumable output
    """

    pass
