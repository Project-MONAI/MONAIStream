===============
Developer Guide
===============

MONAI Stream pipelines are componsed of three types of components: `sources`, `filters` and `sinks`, each of which can
be distinguished from the other by type as they are respectively represented by :py:class:`~monaistream.interface.StreamSourceComponent`,
:py:class:`~monaistream.interface.StreamFilterComponent`, and :py:class:`~monaistream.interface.StreamSinkComponent`.

In the next section we provide a summary of the most commonly used MONAI Stream components, then we elaborate on
how they can be combined together to create streaming inference apps.

MONAI Stream Pipeline components
================================

:py:class:`~monaistream.sources.uri.URISource`
----------------------------------------------

``URISource`` is a source component which generates data when provided with a URI to a data source such as a local video file,
a video stream (e.g. via ``http://``), or a live stream.

.. code-block:: python

   URISource(uri="file:///my/videos/video.mp4"),

.. note::
   
   Feed from multiple ``URISource`` components can be aggregated together via ``NVAggregatedSourcesBin`` (see below).

:py:class:`~monaistream.sources.ajavideosrc.AJAVideoSource`
-----------------------------------------------------------

``AJAVideoSource`` provides support for live streaming from AJA capture devices connected to `x86` or `Clara AGX` systems.

.. warning::
   
   ``AJAVideoSource`` is currently not compatible with ``NVAggregatedSourcesBin``.

:py:class:`~monaistream.sources.fake.FakeSource`
------------------------------------------------

``FakeSource`` is a sink component that allows the developer to end the MONAI Stream pipeline without the need to visualize data.

:py:class:`~monaistream.sources.sourcebin.NVAggregatedSourcesBin`
-----------------------------------------------------------------

``NVAggregatedSourcesBin`` is a special type of `source` component, which can aggregate data from multiple sources
and present it as a single unit of data concatenated in the batch dimension. For instance it may concatenate multiple
``URISource`` components as shown below, resizing all to the same dimensions, and stacking the data in the batch dimension.

The code below show an example declaration of ``NVAggregatedSourcesBin`` with two ``URISource`` components, the output of which
is resized to `864 x 480` and stacked in the batch dimensions.

.. code-block:: python

   NVAggregatedSourcesBin(
      [
         URISource(uri="file:///my/videos/video1.mp4"),
         URISource(uri="file:///my/videos/video2.mp4"),
      ],
      output_width=864,
      output_height=480,
   )

.. mermaid::

   stateDiagram-v2
      state "URISource" as URISource_1
      state "URISource" as URISource_2
      state NVAggregatedSourcesBin {
         URISource_1 --> BatchData
         URISource_2 --> BatchData
         BatchData --> [*]
      }

:py:class:`~monaistream.filters.convert.NVVideoConvert`
-------------------------------------------------------

``NVVideoConvert`` is a filter component which allows the developer to convert the upstream data both in format and size.

For example we may want to create an ``NVVideoConvert`` component that converts data to ``RGBA`` with size ``864 x 480``.

.. code-block:: python

   NVVideoConvert(
         FilterProperties(
            format="RGBA",
            width=864,
            height=480,
         )
   )

:py:class:`~monaistream.filters.infer.NVInferServer`
----------------------------------------------------

``NVInferServer`` receives the output of ``NVVideoConvert`` and runs a configured AI model to produce results (e.g. segmentation, classification, etc.)
in the form of `User Metadata`. This means that ``NVInferServer`` outputs primarily the original input along with inference results in user medatadata,
therefore one must be careful to select the correct data in the following component.

For the ``NVInferServer`` the developer will need to specify a configuration using the infer server configuration objects
:py:class:`~monaistream.filters.infer.InferServerConfiguration`. In the example below, ``NVInferServer`` uses the default
configuration with minor modifications specifying the path to the model repository ``/app/models``, the model name ``cholec_unet_864x480``,
the model version (``-1`` referring to the latest), and the inference server log verbosity.

.. code-block:: python

   infer_server_config = NVInferServer.generate_default_config()
   infer_server_config.infer_config.backend.trt_is.model_repo.root = "/app/models"
   infer_server_config.infer_config.backend.trt_is.model_name = "us_unet_256x256"
   infer_server_config.infer_config.backend.trt_is.version = "-1"
   infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0
   
   ...

   NVInferServer(
      config=infer_server_config,
   )

The inference server received the data provided to it from the upstream component (e.g. ``NVVideoConvert``) and performs inference based
on the configured models in the model repo. The results of the inference are stored in the "user metadata", therefore the primary output
of ``NVInferServer`` is the original data stream and the results are stores in the user metadata. we will see how to access the user metadata
in the ``TransformsChainComponent``.

.. mermaid::

   stateDiagram-v2
      state NVInferServer {
         [*] --> Model
         Model --> Model_Output_1
         Model --> Model_Output_...
         Model --> Model_Output_N
         [*] --> [*]
         Model_Output_1 --> User_Metadata[1..N]
         Model_Output_... --> User_Metadata[1..N]
         Model_Output_N --> User_Metadata[1..N]
         User_Metadata[1..N] --> [*]
      }

:py:class:`~monaistream.filters.transform.TransformChainComponent`
------------------------------------------------------------------

``TransformChainComponent`` is a filter component which allows the developer to apply `MONAI transformations <https://docs.monai.io/en/latest/transforms.html#dictionary-transforms>`_ to streaming data coming from
any other MONAI Stream `source` or `filter`. When placed after an ``NVInferServer`` component it takes all the inputs, original and user metadata,
presents them to the MONAI transformations specified in the ```transform_chain``` parameter, and outputs the result
specified by the ``output_label`` parameter. The inputs to the transform chain are labelled as follows:
 
  - the original stream is always present in the inputs with key ``ORIGINAL_IMAGE``,
  - additional inputs to the transform chain are only available when ``TransformChainComponent`` follows ``NVInferServer``
    where the keys to each output from the model in the ``NVInferServer`` match the output names of the model (see code below).

.. warning::

   Currently, ``TransformChainComponent`` has limitations on the size of the input and output. Specifically, the size of the output in the ``transform_chain``
   must be the same as the size of the input.

In the example below, ``TransformChainComponent`` will output the data with key ``output_label="CONCAT_IMAGE"``. Here, the input keys to the ``transform_chain``
are ``"ORIGINAL_IMAGE"`` and ``"OUTPUT__0"``, where the latter is the output label of the model defined in the ``NVInferServer`` in the last section.

.. code-block:: python
   :emphasize-lines: 2, 27

   # define a color-blending function to be used in the transform chain below
   def color_blender(img: torch.Tensor):
      # show background segmentation as red
      img[..., 1] -= img[..., 1] * (1.0 - img[..., 4])
      img[..., 2] -= img[..., 2] * (1.0 - img[..., 4])

      # show foreground segmentation as blue
      img[..., 0] -= img[..., 0] * img[..., 5]
      img[..., 1] -= img[..., 1] * img[..., 5]

      return img[..., :4]

   ...

   TransformChainComponent(
      # choose the label in the transform chain which we want to output
      output_label="CONCAT_IMAGE",
      # specify transformation to be applied to data
      transform_chain=Compose(
         [
            # apply post-transforms to segmentation model output `OUTPUT__0`
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
   )

.. mermaid::

   stateDiagram-v2
      state TransformChainComponent {
         [*] --> ImplicitInputMapping
         state "CastToTyped" as CastToTypedFLOAT
         state "CastToTyped" as CastToTypedINT
         state ImplicitInputMapping {
            state "[ Input[0], Input[1] ]" as IMInputs
            state "{<br>'ORIGINAL_IMAGE': Input[0],<br> 'OUTPUT__0': Input[1]<br>}" as IMOutputs
            [*] --> IMInputs
            IMInputs --> IMOutputs: Map List to Dict
            IMOutputs --> [*]
         }
         ImplicitInputMapping --> Activationsd
         Activationsd --> AsDiscreted
         AsDiscreted --> AsChannelLastd
         AsChannelLastd --> CastToTypedFLOAT
         CastToTypedFLOAT --> ConcatItemsd
         ConcatItemsd --> Lambdad
         Lambdad --> ScaleIntensityd
         ScaleIntensityd --> CastToTypedINT
         CastToTypedINT --> ImplicitOutputMapping
         state ImplicitOutputMapping {
               state "{<br>'ORIGINAL_IMAGE': Output[0],<br> 'OUTPUT__0': Output[1],<br>'CONCAT_IMAGE': Output[2]<br/>}" as OMInputs
               state "Output[2]" as OMOutputs
               [*] --> OMInputs
               OMInputs --> OMOutputs: Select "CONCAT_IMAGE"
               OMOutputs --> [*]
         }
         ImplicitOutputMapping --> [*]
      }

:py:class:`~monaistream.filters.transform_cupy.TransformChainComponentCupy`
---------------------------------------------------------------------------

``TransformChainComponentCupy`` is a filter component which allows the developer to insert custom data transformations that employ Cupy.
It is a temporary counterpart to ``TransformChainComponent`` for use mainly in applications expected to run in `Clara AGX` devices as
PyTorch (and by extension `MONAI SDK <https://github.com/Project-MONAI/MONAI>`_) is currently not supported in `Clara AGX` devices.

``TransformChainComponentCupy`` works the same fashion as ``TransformChainComponent``, however, it passes ``Dict[str, cupy.ndarray]``
to the ``transform_chain``.

.. code-block:: python

   # color blender function used in `TransformChainComponentCupy`
   def color_blender(inputs: Dict[str, cupy.ndarray]):
      img = inputs["ORIGINAL_IMAGE"]
      mask = inputs["OUTPUT__0"]

      mask = cupy.cudnn.activation_forward(mask, cupy.cuda.cudnn.CUDNN_ACTIVATION_SIGMOID)

      # Ultrasound model outputs two channels, so modify only the red
      # and green channel in-place to apply mask.
      img[..., 1] = cupy.multiply(cupy.multiply(mask[0, ...], 1.0 - mask[1, ...]), img[..., 1])
      img[..., 2] = cupy.multiply(mask[0, ...], img[..., 2])
      img[..., 0] = cupy.multiply(1.0 - mask[1, ...], img[..., 0])

      return {"BLENDED_IMAGE": img}

   ...

   # we select the "BLENDED_IMAGE" output from `color_blender`
   TransformChainComponentCupy(transform_chain=color_blender, output_label="BLENDED_IMAGE"),

:py:class:`~monaistream.sinks.nveglglessink.NVEglGlesSink`
----------------------------------------------------------

``NVEglGlesSink`` is a component that allows developers to visualize the outputs of their pipelines when data is streamed via NVIDIA GPU.

:py:class:`~monaistream.sinks.fake.FakeSink`
--------------------------------------------

``FakeSink`` is a sink component that allows the developer to end the MONAI Stream pipeline without the need to visualize data. ``FakeSink``
is useful for unit testing and for cases where ``TransformChainComponent`` outputs data to disk, but provides no output other than the original
data stream.


MONAI Stream Pipelines by Example
=================================

A MONAI Stream pipeline is a chain composition of MONAI Stream components that begins with one or more 
``StreamSourceComponent``, ends with ``StreamSinkComponent``, and in between uses ``StreamFilterComponent``
to manipulate the data such as applying transformations and running AI inference.

MONAI Stream with Aggregated Sources
------------------------------------

Let us walk through a simple example such as `monaistream-pytorch-pp-app <LINKREF_GITHUB_MONAISTREAM/sample/monaistream-pytorch-pp-app/main.py>`_
where the pipeline can be visualized as shown below.

.. mermaid::

  stateDiagram-v2
      NVAggregatedSourcesBin --> NVVideoConvert: BatchData Output
      NVVideoConvert --> NVInferServer: RGBA Output
      NVInferServer --> TransformChainComponent: ORIGINAL_IMAGE
      NVInferServer --> TransformChainComponent: OUTPUT__0
      TransformChainComponent --> NVEglGlesSink: CONCAT_IMAGE

We can create this streaming inference app using the following code.

.. code-block:: python
   :linenos:

   # generate a default configuration for `NVInferServer`
   infer_server_config = NVInferServer.generate_default_config()
   
   # update default configuration with 
   #   - model repo path
   #   - model name
   #   - model version
   #   - NVInferServer log verbosity
   infer_server_config.infer_config.backend.trt_is.model_repo.root = "/app/models"
   infer_server_config.infer_config.backend.trt_is.model_name = "us_unet_256x256"
   infer_server_config.infer_config.backend.trt_is.version = "-1"
   infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0

   # simple color blender function to use in `Lambdad` MONAI transform
   def color_blender(img: torch.Tensor):
      # show background segmentation as red
      img[..., 1] -= img[..., 1] * (1.0 - img[..., 4])
      img[..., 2] -= img[..., 2] * (1.0 - img[..., 4])

      # show foreground segmentation as blue
      img[..., 0] -= img[..., 0] * img[..., 5]
      img[..., 1] -= img[..., 1] * img[..., 5]

      return img[..., :4]

   pipeline = StreamCompose(
      [
         # read from local video file using `URISource` and use
         # `NVAggregatedSourcesBin` to apply sizing transformations
         NVAggregatedSourcesBin(
               [
                  URISource(uri="file:///app/videos/Q000_04_tu_segmented_ultrasound_256.avi"),
               ],
               output_width=256,
               output_height=256,
         ),
         # convert video stream to RGBA
         NVVideoConvert(
               FilterProperties(
                  format="RGBA",
                  width=256,
                  height=256,
               )
         ),
         # chain output to `NVInferServer`
         NVInferServer(
               config=infer_server_config,
         ),
         # use `TransformChainComponent` to blend the original image with the segmentation
         # output from `NVInferServer`
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
         # display output for `TransformChainComponent`
         NVEglGlesSink(sync=True),
      ]
   )

   # execute pipeline
   pipeline()

Looking more closely at lines `9` and `10`, the ``NVInferServer`` component expects a model named ``us_unet_256x256``
under ``/app/models`` with the `directory structure <https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout>`_ 
and model `configuration <https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md>`_ file
expected by the `Triton Inference Server <https://github.com/triton-inference-server/server>`_.

In this case the directory structure of the model required the pipeline above is

.. code-block::

   /app/models
   └── us_unet_256x256
      ├── 1
      │   └── monai_unet.engine
      └── config.pbtxt

and the model configuration file ``config.pbtxt`` describes the model metadata, as below.

.. code-block::
   :emphasize-lines: 14

   name: "us_unet_256x256"
   platform: "tensorrt_plan"
   default_model_filename: "monai_unet.engine"
   max_batch_size: 1
   input [
      {
         name: "INPUT__0"
         data_type: TYPE_FP32
         dims: [ 3, 256, 256 ]
      }
   ]
   output [
      {
         name: "OUTPUT__0"
         data_type: TYPE_FP32
         dims: [ 2, 256, 256]
      }
   ]

   # Specify GPU instance.
   instance_group {
      kind: KIND_GPU
      count: 1
      gpus: 0
   }

The model configuration file specifies the model type as a TensorRT plan, and it's expected inputs and outputs. The
highlighted line in the model configuration shows the (one and only in this case) model output ``OUTPUT__0`` that
will be passed from ``NVInferServer`` to ``TransformChainComponent``. Following the pipeline code snippet above
it is apparent that the label ``OUTPUT__0`` of the model configuration matches the key of the object being manipulated
in the ``transform_chain`` in line `53`.

AJA Video Capture app
---------------------

MONAI Stream provides native support for AJA capture cards with GPU direct memory access. A simple example is provided
in `monaistream-rdma-capture-app <LINKREF_GITHUB_MONAISTREAM/sample/monaistream-rdma-capture-app/main.py>`_
where the pipeline can be visualized as shown below.

.. mermaid::
   
   stateDiagram-v2
      AJAVideoSource --> NVEglGlesSink: RGBA 1080p in GPU

The visualized pipeline is built using the code below.

.. code-block:: python

   # create a MONAI Stream pipeline for AJA capture with GPU RDMA
   pipeline = StreamCompose(
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

   # start pipeline
   pipeline()

While the pipeline here is simple, developers can add ``NVInferServer`` and ``TransformChainComponent`` to perform
live streaming inference using AJA video capture cards on `x86` or `Clara AGX`.
