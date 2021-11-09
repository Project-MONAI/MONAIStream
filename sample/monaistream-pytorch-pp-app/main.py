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
from monai.transforms.utility.dictionary import AsChannelLastd, CastToTyped, ConcatItemsd

from monaistream.compose import StreamCompose
from monaistream.filters import FilterProperties, NVInferServer, NVVideoConvert
from monaistream.filters.transform import TransformChainComponent
from monaistream.sinks import NVEglGlesSink
from monaistream.sources import NVAggregatedSourcesBin, URISource

logging.basicConfig(level=logging.DEBUG)


def color_blender(img: torch.Tensor):
    # show background segmentation as red
    img[..., 1] -= img[..., 1] * (1. - img[..., 4])
    img[..., 2] -= img[..., 2] * (1. - img[..., 4])

    # show foreground segmentation as blue
    img[..., 0] -= img[..., 0] * img[..., 5]
    img[..., 1] -= img[..., 1] * img[..., 5]

    return img[..., :4]


if __name__ == "__main__":

    infer_server_config = NVInferServer.generate_default_config()
    infer_server_config.infer_config.backend.trt_is.model_repo.root = "/app/models"
    infer_server_config.infer_config.backend.trt_is.model_name = "us_unet_256x256"
    infer_server_config.infer_config.backend.trt_is.version = "1"
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
            TransformChainComponent(
                output_label="CONCAT_IMAGE",
                transform_chain=Compose(
                    [
                        # apply post-transforms to segmentation
                        Activationsd(keys=["OUTPUT__0"], sigmoid=True),
                        AsDiscreted(keys=["OUTPUT__0"]),
                        AsChannelLastd(keys=["OUTPUT__0"]),
                        # concatenate segmentation and original image
                        CastToTyped(keys=["ORIGINAL_IMAGE"], dtype=np.float),
                        ConcatItemsd(keys=["ORIGINAL_IMAGE", "OUTPUT__0"], name="CONCAT_IMAGE", dim=2),
                        # blend the original image and segmentation
                        Lambdad(keys=["CONCAT_IMAGE"], func=color_blender),
                        ScaleIntensityd(keys=["CONCAT_IMAGE"], minv=0, maxv=256),
                        CastToTyped(keys=["CONCAT_IMAGE"], dtype=np.uint8),
                    ]
                ),
            ),
            NVEglGlesSink(sync=True),
        ]
    )
    chain()
