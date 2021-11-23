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

import os
import unittest
from typing import Dict

import cupy
import torch

from monaistream.compose import StreamCompose
from monaistream.filters import FilterProperties, NVVideoConvert, TransformChainComponent, TransformChainComponentCupy
from monaistream.filters.infer import NVInferServer
from monaistream.sinks.nveglglessink import NVEglGlesSink
from monaistream.sources.sourcebin import NVAggregatedSourcesBin
from monaistream.sources.uri import URISource


class TestWithData(unittest.TestCase):
    def test_customuserdata(self):
        def assert_copy_equal(inputs: Dict[str, torch.Tensor]):
            self.assertEqual(inputs["ORIGINAL_IMAGE"], inputs["OUTPUT0"])
            return inputs

        infer_server_config = NVInferServer.generate_default_config()
        infer_server_config.infer_config.backend.trt_is.model_repo.root = os.path.join(os.getcwd(), "tests", "models")
        infer_server_config.infer_config.backend.trt_is.model_name = "identity"
        infer_server_config.infer_config.backend.trt_is.version = "1"
        infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0

        pipeline = StreamCompose(
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
                TransformChainComponent(
                    output_label="ORIGINAL_IMAGE",
                    transform_chain=assert_copy_equal,
                ),
                NVEglGlesSink(sync=False),
            ]
        )
        pipeline()

    def test_customuserdatacupy(self):
        def assert_copy_equal(inputs: Dict[str, cupy.ndarray]):
            self.assertEqual(inputs["ORIGINAL_IMAGE"], inputs["OUTPUT0"])
            return inputs

        infer_server_config = NVInferServer.generate_default_config()
        infer_server_config.infer_config.backend.trt_is.model_repo.root = os.path.join(os.getcwd(), "tests", "models")
        infer_server_config.infer_config.backend.trt_is.model_name = "identity"
        infer_server_config.infer_config.backend.trt_is.version = "1"
        infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0

        pipeline = StreamCompose(
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
                TransformChainComponentCupy(
                    output_label="ORIGINAL_IMAGE",
                    transform_chain=assert_copy_equal,
                ),
                NVEglGlesSink(sync=False),
            ]
        )
        pipeline()


if __name__ == "__main__":
    t = TestWithData()
    t.test_customuserdata()
