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

from monai.transforms import Identityd, Compose

from monaistream.sources import FakeSource
from monaistream.sinks import FakeSink
from monaistream.compose import StreamCompose
from monaistream.filters import NVVideoConvert, FilterProperties, TransformChainComponent


class TestRBGAWithFake(unittest.TestCase):

    def test_shortcircuit(self):
        pipeline = StreamCompose(
            [
                FakeSource(),
                FakeSink(),
            ]
        )
        pipeline()

    def test_identitytransformchain(self):
        pipeline = StreamCompose(
            [
                FakeSource(),
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
                FakeSource(),
                NVVideoConvert(
                    FilterProperties(
                        format="RGBA",
                        width=1920,
                        height=1080,
                        channels=4,
                        framerate=(32, 1),
                    )
                ),
                FakeSink(),
            ]
        )
        pipeline()