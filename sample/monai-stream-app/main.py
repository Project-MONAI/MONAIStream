# from monai.transforms.compose import Compose
# from monai.transforms.intensity.array import NormalizeIntensity
from stream.compose import StreamCompose
from stream.filters.convert import NVRGBAFilter, NVVideoConvert
from stream.filters.infer import NVInferServer
from stream.filters.nvstreammux import NVStreamMux
# from stream.filters.transform import TransformChainComponent
from stream.sinks import NVEglGlesSink
from stream.sources import NVAggregatedSourcesBin, URISource

if __name__ == "__main__":

    # pre_transforms = TransformChainComponent(
    #     Compose([
    #         NormalizeIntensity()
    #     ])
    # )

    inferServerConfig = NVInferServer.generate_default_config()
    inferServerConfig.infer_config.backend.trt_is.model_repo.root = "/app/models"
    inferServerConfig.infer_config.backend.trt_is.model_name = "monai_unet_pytorch"
    inferServerConfig.infer_config.backend.trt_is.version = "-1"
    inferServerConfig.infer_config.backend.trt_is.model_repo.log_level = 0
    inferServer = NVInferServer(config=inferServerConfig)

    chain = StreamCompose([
        NVAggregatedSourcesBin([
            URISource(uri="file:///app/videos/d1_im.mp4"),
        ]),
        NVStreamMux(
            num_sources=1,
            width=1260,
            height=1024,
        ),
        NVVideoConvert(),
        NVRGBAFilter(),
        # pre_transforms,
        inferServer,
        NVEglGlesSink(
            sync=True,
        ),
    ])
    chain()
