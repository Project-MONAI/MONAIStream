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

import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from gi.repository import Gst
from jinja2 import Template
from pydantic import BaseModel
from typing_extensions import Literal

from monaistream.errors import BinCreationError
from monaistream.interface import InferenceFilterComponent


class TritonModelRepo(BaseModel):
    root: str = "."
    log_level: Optional[int]
    strict_model_config: bool = True


class TrtISParams(BaseModel):
    model_name: str = ""
    version: int = -1
    model_repo: TritonModelRepo


class IOLayer(BaseModel):
    name: str


class OutputLayer(IOLayer):
    pass


class InputLayer(IOLayer):
    dims: List[int]
    data_type: Literal[
        "TENSOR_DT_NONE",
        "TENSOR_DT_FP32",
        "TENSOR_DT_FP16",
        "TENSOR_DT_INT8",
        "TENSOR_DT_INT16",
        "TENSOR_DT_INT32",
        "TENSOR_DT_UINT8",
        "TENSOR_DT_UINT16",
        "TENSOR_DT_UINT32",
    ] = "TENSOR_DT_NONE"


class BackendParams(BaseModel):
    inputs: Optional[List[InputLayer]]
    outputs: Optional[List[OutputLayer]]
    trt_is: TrtISParams


class NormalizeModel(BaseModel):
    scale_factor: float
    channel_offsets: Optional[List[int]]


class PreprocessParams(BaseModel):
    network_format: Literal[
        "MEDIA_FORMAT_NONE",
        "IMAGE_FORMAT_RGB",
        "IMAGE_FORMAT_BGR",
        "IMAGE_FORMAT_GRAY",
    ] = "IMAGE_FORMAT_RGB"
    tensor_order: Literal["TENSOR_ORDER_NONE", "TENSOR_ORDER_LINEAR", "TENSOR_ORDER_NHWC"] = "TENSOR_ORDER_LINEAR"
    tensor_name: Optional[str]
    maintain_aspect_ratio: Literal[0, 1] = 0
    frame_scaling_hw: Literal[
        "FRAME_SCALING_HW_DEFAULT", "FRAME_SCALING_HW_GPU", "FRAME_SCALING_HW_VIC"
    ] = "FRAME_SCALING_HW_DEFAULT"
    frame_scaling_filter: int = 1
    normalize: Optional[NormalizeModel]


class PostprocessParams(BaseModel):
    other: Dict[str, str] = {}


class ExtraControl(BaseModel):
    copy_input_to_host_buffers: bool = False
    output_buffer_pool_size: Optional[int]


class CustomLib(BaseModel):
    path: str


class InferenceConfig(BaseModel):
    unique_id: int
    gpu_ids: List[int] = [0]
    max_batch_size: int = 1
    backend: BackendParams
    preprocess: PreprocessParams
    postprocess: PostprocessParams
    custom_lib: Optional[CustomLib]
    extra: ExtraControl


class BBoxFilter(BaseModel):
    min_width: int = 64
    min_height: int = 64
    max_width: int = 640
    max_height: int = 640


class InputObjectControl(BaseModel):
    bbox_filter: BBoxFilter


class InputControl(BaseModel):
    process_mode: Literal["PROCESS_MODE_FULL_FRAME", "PROCESS_MODE_CLIP_OBJECTS"] = "PROCESS_MODE_FULL_FRAME"
    operate_on_gie_id: Optional[int]
    operate_on_class_ids: Optional[List[int]]
    interval: int = 0
    async_mode: Optional[bool]
    object_control: Optional[InputObjectControl]


class Color(BaseModel):
    r: float
    g: float
    b: float
    a: float


class DetectClassFilter(BaseModel):
    bbox_filter: BBoxFilter
    roi_top_offset: int
    roi_bottom_offset: int
    border_color: Color
    bg_color: Color


class OutputObjectDetectionControl(BaseModel):
    default_filter: DetectClassFilter
    specific_class_filters: Optional[Dict[int, DetectClassFilter]]


class OutputControl(BaseModel):
    output_tensor_meta: bool = True
    detect_control: Optional[OutputObjectDetectionControl]


class InferServerConfiguration(BaseModel):
    infer_config: InferenceConfig
    input_control: InputControl
    output_control: OutputControl
    process_mode: Optional[Literal["PROCESS_MODE_FULL_FRAME", "PROCESS_MODE_CLIP_OBJECTS"]] = "PROCESS_MODE_FULL_FRAME"
    operate_on_gie_id: Optional[int]
    operate_on_class_ids: Optional[List[int]]
    interval: Optional[int]
    async_mode: Optional[bool]
    object_control: Optional[InputObjectControl]


class NVInferServer(InferenceFilterComponent):
    """
    Triton Inference server component
    """

    output_template = """infer_config {
    unique_id: {{ infer_config.unique_id }}
    {%- if infer_config.gpu_ids is defined and infer_config.gpu_ids is not none %}
    gpu_ids: [{%- for id in infer_config.gpu_ids -%} {{ id }}{{ "," if not loop.last else "" }} {%- endfor -%}]
    {%- endif %}
    max_batch_size: {{ infer_config.max_batch_size|default(4) }}
    backend {
        trt_is {
            model_name: "{{ infer_config.backend.trt_is.model_name }}"
            version: {{ infer_config.backend.trt_is.version|default(-1) }}
            model_repo {
                root: "{{ infer_config.backend.trt_is.model_repo.root|default(".") }}"
                strict_model_config: {{ infer_config.backend.trt_is.model_repo.strict_model_config|default(true)|string|lower }}
                {%- if infer_config.backend.trt_is.model_repo.log_level is defined and infer_config.backend.trt_is.model_repo.log_level is not none %}
                log_level: {{ infer_config.backend.trt_is.model_repo.log_level }}
                {%- endif %}
            }
        }
        {%- if infer_config.backend.inputs is defined and infer_config.backend.inputs is not none %}
        inputs [
            {%- for input in infer_config.backend.inputs %}
            {
                name: "{{ input.name }}"
                dims: [ {%- for dim in input.dims -%} {{ dim }}{{ "," if not loop.last else "" }} {%- endfor -%}]
                data_type: {{ input.data_type|default("TENSOR_DT_NONE") }}
            }
            {%- endfor %}
        ]
        {%- endif %}
        {%- if infer_config.backend.outputs is defined and infer_config.backend.outputs is not none %}
        outputs [
            {%- for output in infer_config.backend.outputs %}
            {
                name: "{{ output.name }}"
            }
            {%- endfor %}
        ]
        {%- endif %}
    }

    preprocess {
        network_format: {{ infer_config.preprocess.network_format|default("IMAGE_FORMAT_RGB") }}
        tensor_order: {{ infer_config.preprocess.tensor_order|default("TENSOR_ORDER_LINEAR") }}
        {%- if infer_config.preprocess.tensor_name is defined and infer_config.preprocess.tensor_name is not none %}
        tensor_name: {{ infer_config.preprocess.tensor_name }}
        {%- endif %}
        maintain_aspect_ratio: {{ infer_config.preprocess.maintain_aspect_ratio|default(0) }}
        frame_scaling_hw: {{ infer_config.preprocess.frame_scaling_hw|default("FRAME_SCALING_HW_DEFAULT") }}
        frame_scaling_filter: {{ infer_config.preprocess.frame_scaling_filter|default(1) }}
        {%- if infer_config.preprocess.normalize is defined and infer_config.preprocess.normalize is not none %}
        normalize {
            scale_factor: {{ infer_config.preprocess.normalize.scale_factor|default(0.00392156) }}
            {% if infer_config.preprocess.normalize.channel_offsets is defined and infer_config.preprocess.normalize.channel_offsets is not none -%}
            channel_offsets: [{%- for offset in infer_config.preprocess.normalize.channel_offsets -%} {{offset}}{{ "," if not loop.last else "" }} {% endfor -%}]
            {%- endif %}
        }
        {% else %}
        normalize {
            scale_factor: 0.00392156,
        }
        {%- endif %}
    }

    postprocess {
        other {
            {%- for k, v in infer_config.postprocess.other.items() %}
            {{ k }}: {{ v }}
            {%- endfor %}
        }
    }

    extra {
        copy_input_to_host_buffers: {{ infer_config.extra.copy_input_to_host_buffers|default(false)|string|lower }}
        {% if infer_config.extra.output_buffer_pool_size is defined and infer_config.extra.output_buffer_pool_size is not none %}
        output_buffer_pool_size: {{ infer_config.extra.output_buffer_pool_size }}
        {%- endif %}
    }
    {%- if infer_config.custom_lib is defined and infer_config.custom_lib is not none %}
    custom_lib {
        path: {{ infer_config.custom_lib.path }}
    }
    {%- endif %}
}

input_control {
    process_mode: {{ input_control.process_mode|default("PROCESS_MODE_FULL_FRAME") }}
    {%- if input_control.operate_on_gie_id is defined and input_control.operate_on_gie_id is not none %}
    operate_on_gie_id: {{ input_control.operate_on_gie_id }}
    {%- endif %}
    {%- if input_control.operate_on_class_ids is defined and input_control.operate_on_class_ids is not none %}
    operate_on_class_ids: [{% for clsid in input_control.operate_on_class_ids -%} {{ clsid }}{{ "," if not loop.last else "" }} {%- endfor %}]
    {%- endif %}
    interval: {{ input_control.interval|default(0) }}
    {%- if input_control.async_mode is defined and input_control.async_mode is not none %}
    async_mode: {{ input_control.async_mode }}
    {%- endif %}
    {%- if input_control.object_control is defined and input_control.object_control is not none %}
    object_control {
        bbox_filter {
            min_width: {{ input_control.object_control.bbox_filter.min_width }}
            min_height: {{ input_control.object_control.bbox_filter.min_height }}
            max_width: {{ input_control.object_control.bbox_filter.max_width }}
            max_height: {{ input_control.object_control.bbox_filter.max_height }}
        }
    }
    {%- endif %}
}

output_control {
    output_tensor_meta: {{ output_control.output_tensor_meta|default(true) }}
    {%- if output_control.detect_control is defined and output_control.detect_control is not none %}
    detect_control {
        default_filter {
            bbox_filter {
                min_width: {{ output_control.detect_control.default_filter.bbox_filter.min_width }}
                min_height: {{ output_control.detect_control.default_filter.bbox_filter.min_height }}
                max_width: {{ output_control.detect_control.default_filter.bbox_filter.max_width }}
                max_height: {{ output_control.detect_control.default_filter.bbox_filter.max_height }}
            }
            roi_top_offset: {{ output_control.detect_control.roi_top_offset }}
            roi_bottom_offset: {{ output_control.detect_control.roi_bottom_offset }}
            border_color {
                r: {{ output_control.detect_control.default_filter.border_color.r }}
                g: {{ output_control.detect_control.default_filter.border_color.g }}
                b: {{ output_control.detect_control.default_filter.border_color.b }}
                a: {{ output_control.detect_control.default_filter.border_color.a }}
            }
            bg_color {
                r: {{ output_control.detect_control.default_filter.bg_color.r }}
                g: {{ output_control.detect_control.default_filter.bg_color.g }}
                b: {{ output_control.detect_control.default_filter.bg_color.b }}
                a: {{ output_control.detect_control.default_filter.bg_color.a }}
            }
        }
        {%- if output_control.detect_control.specific_class_filters is defined and output_control.detect_control.specific_class_filters is not none %}
        specific_class_filters {
            {%- for k, dcf in output_control.detect_control.specific_class_filters.items() %}
            {{ k }} {
                bbox_filter {
                    min_width: {{ dcf.bbox_filter.min_width }}
                    min_height: {{ dcf.bbox_filter.min_height }}
                    max_width: {{ dcf.bbox_filter.max_width }}
                    max_height: {{ dcf.bbox_filter.max_height }}
                }
                roi_top_offset: {{ dcf.roi_top_offset }}
                roi_bottom_offset: {{ dcf.roi_bottom_offset }}
                border_color {
                    r: {{ dcf.border_color.r }}
                    g: {{ dcf.border_color.g }}
                    b: {{ dcf.border_color.b }}
                    a: {{ dcf.border_color.a }}
                }
                bg_color {
                    r: {{ dcf.bg_color.r }}
                    g: {{ dcf.bg_color.g }}
                    b: {{ dcf.bg_color.b }}
                    a: {{ dcf.bg_color.a }}
                }
            }
            {%- endfor %}
        }
        {%- endif %}
    }
    {%- endif %}
}

{% if operate_on_gie_id is defined and operate_on_gie_id is not none -%}
operate_on_gie_id: {{ operate_on_gie_id }}
{%- endif %}

{% if operate_on_class_ids is defined and operate_on_class_ids is not none -%}
operate_on_class_ids: [{%- for id in operate_on_class_ids -%} {{ id }}{{ "," if not loop.last else "" }} {%- endfor -%}]
{%- endif %}

{% if interval is defined and interval is not none -%}
interval: {{ interval }}
{%- endif %}

{% if async_mode is defined and async_mode is not none -%}
async_mode: {{ async_mode|string|lower }}
{%- endif %}
"""

    default_config = """
    {
        "infer_config": {
            "unique_id": 1,
            "gpu_ids": [0],
            "max_batch_size": 1,
            "backend": {
                "trt_is": {
                    "model_name": "",
                    "version": -1,
                    "model_repo": {
                        "root": ".",
                        "strict_model_config": true
                    }
                }
            },

            "preprocess": {
                "network_format": "IMAGE_FORMAT_RGB",
                "tensor_order": "TENSOR_ORDER_LINEAR",
                "maintain_aspect_ratio": 0,
                "frame_scaling_hw": "FRAME_SCALING_HW_DEFAULT",
                "frame_scaling_filter": 1,
                "normalize": {
                    "scale_factor": 0.00392156
                }
            },

            "postprocess": {
                "other": {
                }
            },

            "extra": {
                "copy_input_to_host_buffers": false
            }

        },

        "input_control": {
            "process_mode": "PROCESS_MODE_FULL_FRAME",
            "interval": 0
        },

        "output_control": {
            "output_tensor_meta": true
        }
    }
    """

    def __init__(
        self,
        name: str = "",
        config: Optional[InferServerConfiguration] = None,
        config_path: str = "/tmp",
    ) -> None:
        """
        Constructor for Triton Inference server component

        :param config: the configuration (:class:`.InferServerConfiguration`) for the Triton Inference Server streaming component
                       (if none is provided a default configuration is used)
        :param name: the name of the component
        """

        if not name:
            self._name = str(uuid4().hex)

        self._config = config
        if not config:
            self._config = NVInferServer.generate_default_config()

        self._tm = Template(NVInferServer.output_template)
        self._config_path = config_path

    @staticmethod
    def generate_default_config():
        """
        Get the default configuration for the Triton Inference server for customization purposes

        :return: the default inference server component of type :class:`.InferServerConfiguration`
        """
        return InferServerConfiguration(**json.loads(NVInferServer.default_config))

    def initialize(self):
        """
        Initialize the `nvinferserver` GStreamer element and configure based on the provided configuration
        """

        self._config_path = os.path.join(self._config_path, f"config-{self.get_name()}.txt")

        with open(self._config_path, "w") as f:
            f.write(self._tm.render(**self._config.dict()))

        pgie = Gst.ElementFactory.make("nvinferserver", self.get_name())
        if not pgie:
            raise BinCreationError(f"Could not create {self.__class__.__name__}")

        self._pgie = pgie
        self._pgie.set_property("config-file-path", self._config_path)

    def get_config(self) -> Any:
        """
        Get the configuration of the component

        :return: the configuration of the Triton Inference server component
        """
        return self._config

    def get_name(self) -> Any:
        """
        Get the name of the component

        :return: the name of the component as `str`
        """
        return f"{self._name}-inference"

    def set_batch_size(self, batch_size: int):
        """
        Configure the batch size of the inference server

        :param batch_size: a positive integer determining the maximum batch size for the inference server
        """
        self._pgie.set_property("batch-size", batch_size)

    def get_gst_element(self):
        """
        Get the `nvinferserver` GStreamer element being wrapped by this component

        :return: the `nvinferserver` GStreamer element
        """
        return (self._pgie,)
