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
   infer_server_config.infer_config.backend.trt_is.model_name = "cholec_unet_864x480"
   infer_server_config.infer_config.backend.trt_is.version = "-1"
   infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0
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
presents them to the MONAI transformations specified in the `transform_chain` parameter as dictated by the `input_labels` parameter, and outputs the result
specified byt he `output_label` parameter.

.. warning::

   Currently, ``TransformChainComponent`` has limitations on the size of the input and output. Specifically, the size of the output in the ``transform_chain``
   must be the same as the size of the input.

In the example below, ``TransformChainComponent`` is initialized to assign the labels ``input_labels=["original_image", "seg_output"]`` to the inputs received in sequence,
and will output the data with key ``output_label="seg_overlay"``. The ``transform_chain`` callables are therefore presented with a ``Dict[str, torch.Tensor]``.

.. code-block:: python

   TransformChainComponent(
      # assign keys to the inputs to `transform_chain`
      input_labels=["original_image", "seg_output"],
      # choose the desired output of `transform_chain`
      output_label="seg_overlay",
      # specify transformation to be applied to data
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
         ]
      ),
   )

.. mermaid::

   stateDiagram-v2
      state TransformChainComponent {
         [*] --> ImplicitInputMapping
         state ImplicitInputMapping {
            state "[ Input[0], Input[1] ]" as IMInputs
            state "{<br>'original_image': Input[0],<br> 'seg_output': Input[1]<br>}" as IMOutputs
            [*] --> IMInputs
            IMInputs --> IMOutputs
            IMOutputs --> [*]
         }
         ImplicitInputMapping --> Activationsd
         Activationsd --> AsDiscreted
         AsDiscreted --> AddChanneld
         AddChanneld --> AsChannelLastd
         AsChannelLastd --> ConcatItemsd
         ConcatItemsd --> Lambdad
         Lambdad --> CastToTyped
         CastToTyped --> ImplicitOutputMapping
         state ImplicitOutputMapping {
               state "{<br>'original_image': Output[0],<br> 'seg_output': Output[1],<br>'seg_overlay': Output[2]<br/>}" as OMInputs
               state "Output[3]" as OMOutputs
               [*] --> OMInputs
               OMInputs --> OMOutputs
               OMOutputs --> [*]
         }
         ImplicitOutputMapping --> [*]
      }

:py:class:`~monaistream.filters.transform_cupy.TransformChainComponentCupy`
---------------------------------------------------------------------------

``TransformChainComponentCupy`` is a filter component which allows the developer to insert custom data transformations that employ Cupy.
It is a temporary counterpart to ``TransformChainComponent`` for use mainly in applications expected to run in `Clara AGX` devices as
PyTorch (and by extension `MONAI SDK <https://github.com/Project-MONAI/MONAI>`_) is currently not supported in `Clara AGX` devices.

``TransformChainComponentCupy`` works in a very similar fashion to ``TransformChainComponent``, however, it does not utilize a ``Dict`` structure
to pass inputs to the ``transform_chain`` and instead presents all inputs to the component as ``List[cupy.ndarray]``.

.. code-block:: python

   def color_blender(img: cupy.ndarray, mask: List[cupy.ndarray]):
      if mask:
         # mask is of range [0,1]
         mask[0] = cupy.cudnn.activation_forward(mask[0], cupy.cuda.cudnn.CUDNN_ACTIVATION_SIGMOID)
         # modify only the red channel in-place to apply mask
         img[..., 0] = cupy.multiply(1.0 - mask[0][0, ...], img[..., 0])
      return

   TransformChainComponentCupy(transform_chain=color_blender)

:py:class:`~monaistream.sinks.nveglglessink.NVEglGlesSink`
----------------------------------------------------------

``NVEglGlesSink`` is a component that allows developers to visualize the outputs of their pipelines when data is streamed via NVIDIA GPU.

:py:class:`~monaistream.sinks.fake.FakeSink`
----------------------------------------------

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
      NVInferServer --> TransformChainComponent: Original Image
      NVInferServer --> TransformChainComponent: User Metadata
      TransformChainComponent --> NVEglGlesSink: <center>Selected Output Key<br>in TransformChainComponent</center>

We can create the streaming inference app by combining the MONAI Stream components as shown below.

.. code-block:: python

   # generate a default configuration for `NVInferServer`
   infer_server_config = NVInferServer.generate_default_config()
   
   # update default configuration with 
   #   - model repo path
   #   - model name
   #   - model version
   #   - NVInferServer log verbosity
   infer_server_config.infer_config.backend.trt_is.model_repo.root = "/app/models"
   infer_server_config.infer_config.backend.trt_is.model_name = "cholec_unet_864x480"
   infer_server_config.infer_config.backend.trt_is.version = "-1"
   infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0
   
   # simple color blender function to use in `Lambdad` MONAI transform
   def color_blender(img: torch.Tensor):
      img[..., 1] = img[..., 4] + img[..., 1] * (1 - img[..., 4])
      return img[..., :4]

   # create a MONAI Stream pipeline
   pipeline = StreamCompose(
       [
         # read from local video file using `URISource` and use
         # `NVAggregatedSourcesBin` to apply sizing transformations
         NVAggregatedSourcesBin(
            [
               URISource(uri="file:///app/videos/endo.mp4"),
            ],
            output_width=864,
            output_height=480,
         ),
         # convert video stream to RGBA
         NVVideoConvert(
            FilterProperties(
               format="RGBA",
               width=864,
               height=480,
            )
         ),
         # chain output to `NVInferServer`
         NVInferServer(
            config=infer_server_config,
         ),
         # use `TransformChainComponent` to blend the original image with the segmentation
         # output from `NVInferServer`
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
               ]
            ),
         ),
         # display output for `TransformChainComponent`
         NVEglGlesSink(sync=True),
       ]
   )

   # execute pipeline
   pipeline()


AJA Video Capture app
---------------------

Let us walk through a simple example such as `monaistream-rdma-capture-app <LINKREF_GITHUB_MONAISTREAM/sample/monaistream-rdma-capture-app/main.py>`_
where the pipeline can be visualized as shown below.

.. mermaid::
   
   stateDiagram-v2
      AJAVideoSource --> NVInferServer: RGBA Output
      NVInferServer --> TransformChainComponent: Original Image
      NVInferServer --> TransformChainComponent: User Metadata
      TransformChainComponent --> NVEglGlesSink: <center>Selected Output Key<br>in TransformChainComponent</center>

We can create the streaming inference app by combining the MONAI Stream components as shown below.

.. code-block:: python

   # simple color blender function using Cupy in-place transformations
   def color_blender(img: cupy.ndarray, mask: List[cupy.ndarray]):
      if mask:
         # mask is of range [0,1]
         mask[0] = cupy.cudnn.activation_forward(mask[0], cupy.cuda.cudnn.CUDNN_ACTIVATION_SIGMOID)
         # modify only the red channel in-place to apply mask
         img[..., 0] = cupy.multiply(1.0 - mask[0][0, ...], img[..., 0])
      return

   # generate a default configuration for `NVInferServer`
   infer_server_config = NVInferServer.generate_default_config()

   # update default configuration with 
   #   - model repo path
   #   - model name
   #   - model version
   #   - NVInferServer log verbosity
   infer_server_config.infer_config.backend.trt_is.model_repo.root = "/app/models"
   infer_server_config.infer_config.backend.trt_is.model_name = "monai_unet_trt"
   infer_server_config.infer_config.backend.trt_is.version = "-1"
   infer_server_config.infer_config.backend.trt_is.model_repo.log_level = 0
   
   # create a MONAI Stream pipeline for AJA RDMA capture
   pipeline = StreamCompose(
      [
         # create an AJA video capture component with GPU RDMA capability
         AJAVideoSource(
            mode="UHDp30-rgba",
            input_mode="hdmi",
            is_nvmm=True,
            output_width=864,
            output_height=480,
         ),
         # perform inference on incoming video stream data
         # and output a segmentation image map
         NVInferServer(
             config=infer_server_config,
         ),
         # blend original input with segmentation output
         TransformChainComponentCupy(transform_chain=color_blender),
         # display image on screen
         NVEglGlesSink(sync=True),
      ]
   )

   # start pipeline
   pipeline()
