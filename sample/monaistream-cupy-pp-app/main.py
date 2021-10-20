import logging
from typing import List

import cupy
import cupy.cuda.cudnn
import cupy.cudnn

from monaistream.compose import StreamCompose
from monaistream.filters import FilterProperties, NVInferServer, NVStreamMux, NVVideoConvert
from monaistream.filters.transform_cupy import TransformChainComponentCupy
from monaistream.sinks import NVEglGlesSink
from monaistream.sources import NVAggregatedSourcesBin, URISource

logging.basicConfig(level=logging.DEBUG)


def color_blender(img: cupy.ndarray, mask: List[cupy.ndarray]):
    if mask:
        # mask is of range [0,1]
        mask[0] = cupy.cudnn.activation_forward(mask[0], cupy.cuda.cudnn.CUDNN_ACTIVATION_SIGMOID)
        # modify only the red channel in-place to apply mask
        img[..., 0] = cupy.multiply(1.0 - mask[0][0, ...], img[..., 0])
    return


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
                    URISource(uri="file:///app/videos/endo.mp4"),
                ]
            ),
            NVStreamMux(
                num_sources=1,
                width=864,
                height=480,
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
            TransformChainComponentCupy(transform_chain=color_blender),
            NVEglGlesSink(sync=True),
        ]
    )
    chain()
