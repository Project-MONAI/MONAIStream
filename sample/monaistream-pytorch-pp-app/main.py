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

import numpy as np
import torch
from monai.transforms import Activationsd, AsDiscreted, Lambdad
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.utility.dictionary import AddChanneld, AsChannelLastd, CastToTyped, ConcatItemsd, RepeatChanneld

from monaistream.compose import StreamCompose
from monaistream.filters import FilterProperties, NVInferServer, NVVideoConvert
from monaistream.filters.transform import TransformChainComponent
from monaistream.sinks import NVEglGlesSink
from monaistream.sources import NVAggregatedSourcesBin, URISource

logging.basicConfig(level=logging.DEBUG)


def color_blender(img: torch.Tensor):
    img[..., 1] = img[..., 4] + img[..., 1] * (1 - img[..., 4])
    return img[..., :4]


if __name__ == "__main__":

    infer_server_config = NVInferServer.generate_default_config()
    infer_server_config.infer_config.backend.trt_is.model_repo.root = "/app/models"
    infer_server_config.infer_config.backend.trt_is.model_name = "cholec_unet_864x480"
    infer_server_config.infer_config.backend.trt_is.version = "-1"
    infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0

    chain = StreamCompose(
        [
            NVAggregatedSourcesBin(
                [
                    URISource(uri="file:///app/videos/endo.mp4"),
                ],
                output_width=864,
                output_height=480,
            ),
            NVVideoConvert(
                FilterProperties(
                    format="RGBA",
                    width=864,
                    height=480,
                )
            ),
            NVInferServer(
                config=infer_server_config,
            ),
            TransformChainComponent(
                input_labels=["original_image", "seg_output"],
                output_label="seg_overlay",
                transform_chain=Compose(
                    [
                        # apply post-transforms to segmentation
                        Activationsd(keys=["seg_output"], sigmoid=True),
                        AsDiscreted(keys=["seg_output"]),
                        AddChanneld(keys=["seg_output"]),
                        AsChannelLastd(keys=["seg_output"]),
                        # merge segmentation and original image into one for viewing
                        ConcatItemsd(keys=["original_image", "seg_output"], name="seg_overlay", dim=2),
                        Lambdad(keys=["seg_overlay"], func=color_blender),
                        CastToTyped(keys="seg_overlay", dtype=np.uint8),
                        # to view segmentation map alone
                        RepeatChanneld(keys=["seg_output"], repeats=4),
                        ScaleIntensityd(keys=["seg_output"], minv=0, maxv=256),
                        CastToTyped(keys="seg_output", dtype=np.uint8),
                    ]
                ),
            ),
            NVEglGlesSink(sync=True),
        ]
    )
    chain()
