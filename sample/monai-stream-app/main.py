import logging

import numpy as np
import torch
from monai.transforms import Activationsd, AsDiscreted, Lambdad
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.utility.dictionary import AddChanneld, AsChannelLastd, CastToTyped, ConcatItemsd, RepeatChanneld

from monaistream.compose import StreamCompose
from monaistream.filters import FilterProperties, NVInferServer, NVStreamMux, NVVideoConvert
from monaistream.filters.transform import TransformChainComponent
from monaistream.sinks import NVEglGlesSink
from monaistream.sources import NVAggregatedSourcesBin, URISource

logging.basicConfig(level=logging.DEBUG)


def color_blender(img: torch.Tensor):
    img[..., 1] = img[..., 4] + img[..., 1] * (1 - img[..., 4])
    return img[..., :4]


if __name__ == "__main__":

    inferServerConfig = NVInferServer.generate_default_config()
    inferServerConfig.infer_config.backend.trt_is.model_repo.root = "/app/models"
    inferServerConfig.infer_config.backend.trt_is.model_name = "monai_unet_trt"
    inferServerConfig.infer_config.backend.trt_is.version = "-1"
    inferServerConfig.infer_config.backend.trt_is.model_repo.log_level = 0

    chain = StreamCompose(
        [
            NVAggregatedSourcesBin(
                [
                    URISource(uri="file:///app/videos/d1_im.mp4"),
                ]
            ),
            NVStreamMux(
                num_sources=1,
                width=1260,
                height=1024,
            ),
            NVVideoConvert(
                FilterProperties(
                    format="RGBA",
                    width=1264,
                    height=1024,
                )
            ),
            NVInferServer(
                config=inferServerConfig,
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
