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

import unittest

from monai.transforms import Compose, Identityd

from monaistream.compose import StreamCompose
from monaistream.filters import FilterProperties, NVVideoConvert, TransformChainComponent
from monaistream.sinks import FakeSink
from monaistream.sources import TestVideoSource


class TestWithFake(unittest.TestCase):
    def test_shortcircuit(self):
        pipeline = StreamCompose(
            [
                TestVideoSource(),
                FakeSink(),
            ]
        )
        pipeline()

    def test_identitytransformchain(self):
        pipeline = StreamCompose(
            [
                TestVideoSource(),
                TransformChainComponent(
                    transform_chain=Compose(
                        Identityd(keys="ORIGINAL_IMAGE"),
                    ),
                    output_label="ORIGINAL_IMAGE",
                ),
                FakeSink(),
            ]
        )
        pipeline()

    def test_nvvideoconvert(self):
        pipeline = StreamCompose(
            [
                TestVideoSource(),
                NVVideoConvert(
                    format_description=FilterProperties(
                        format="RGBA",
                        width=256,
                        height=256,
                        framerate=(32, 1),
                    )
                ),
                FakeSink(),
            ]
        )
        pipeline()
