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

from monaistream.compose import StreamCompose
from monaistream.sinks import NVEglGlesSink
from monaistream.sources import AJAVideoSource

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    chain = StreamCompose(
        [
            AJAVideoSource(
                mode="UHDp30-rgba",
                input_mode="hdmi",
                is_nvmm=True,
                output_width=1920,
                output_height=1080,
            ),
            NVEglGlesSink(sync=True),
        ]
    )
    chain()
