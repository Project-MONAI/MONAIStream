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

import logging
from typing import Dict

import cupy
import cupy.cuda.cudnn
import cupy.cudnn

from monaistream.compose import StreamCompose
from monaistream.filters import FilterProperties, NVInferServer, NVVideoConvert
from monaistream.filters.transform_cupy import TransformChainComponentCupy
from monaistream.sinks import FakeSink
from monaistream.sources import NVAggregatedSourcesBin, URISource

logging.basicConfig(level=logging.ERROR)


def color_blender(inputs: Dict[str, cupy.ndarray]):
    img = inputs["ORIGINAL_IMAGE"]
    mask = inputs["OUTPUT__0"]

    mask = cupy.cudnn.activation_forward(mask, cupy.cuda.cudnn.CUDNN_ACTIVATION_SIGMOID)

    # Ultrasound model outputs two channels, so modify only the red
    # and green channel in-place to apply mask.
    img[..., 1] = cupy.multiply(cupy.multiply(mask[0, ...], 1.0 - mask[1, ...]), img[..., 1])
    img[..., 2] = cupy.multiply(mask[0, ...], img[..., 2])
    img[..., 0] = cupy.multiply(1.0 - mask[1, ...], img[..., 0])

    return {"BLENDED_IMAGE": img}


if __name__ == "__main__":

    infer_server_config = NVInferServer.generate_default_config()
    infer_server_config.infer_config.backend.trt_is.model_repo.root = "/app/models"
    infer_server_config.infer_config.backend.trt_is.model_name = "monai_unet_trt"
    infer_server_config.infer_config.backend.trt_is.version = "-1"
    infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0

    chain = StreamCompose(
        [
            NVAggregatedSourcesBin(
                [
                    URISource(uri="file:///app/videos/Q000_04_tu_segmented_ultrasound_256.avi"),
                ],
                output_width=256,
                output_height=256,
            ),
            NVVideoConvert(
                FilterProperties(
                    format="RGBA",
                    width=256,
                    height=256,
                )
            ),
            NVInferServer(
                config=infer_server_config,
            ),
            TransformChainComponentCupy(transform_chain=color_blender, output_label="BLENDED_IMAGE"),
            FakeSink(),
        ]
    )
    chain()
