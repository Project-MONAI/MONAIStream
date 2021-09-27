import logging
from typing import List

import torch
# from monai.transforms.compose import Compose
# from monai.transforms.intensity.array import NormalizeIntensity
from stream.compose import StreamCompose
from stream.filters import (FilterProperties, NVInferServer, NVStreamMux,
                            NVVideoConvert)
from stream.filters.transform import TransformChainComponent
from stream.sinks import NVEglGlesSink
from stream.sources import NVAggregatedSourcesBin, URISource

if __name__ == "__main__":

    def my_callback(orig: torch.Tensor, data: List[torch.Tensor]):
        logging.info(f"Original data: {orig.size()}")
        for d in data:
            logging.info(f"Additional data: {d.size()}")
        return torch.inverse(orig)

    pre_transforms = TransformChainComponent(
        transform_chain=my_callback,
    )

    # inferServerConfig = NVInferServer.generate_default_config()
    # inferServerConfig.infer_config.backend.trt_is.model_repo.root = "/app/models"
    # inferServerConfig.infer_config.backend.trt_is.model_name = "monai_unet_trt"
    # inferServerConfig.infer_config.backend.trt_is.version = "-1"
    # inferServerConfig.infer_config.backend.trt_is.model_repo.log_level = 0

    chain = StreamCompose([
        NVAggregatedSourcesBin([
            URISource(uri="file:///app/videos/d1_im.mp4"),
        ]),
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
        pre_transforms,
        # NVInferServer(
        #     config=inferServerConfig,
        # ),
        NVEglGlesSink(
            sync=True,
        ),
    ])
    chain()
