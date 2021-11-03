===============
Developer Guide
===============


MONAI Stream Pipeline Overview
==============================

A MONAI Stream pipeline is a chain composition of MONAI Stream components that begins with one or more 
``StreamSourceComponent``, ends with ``StreamSinkComponent``, and in between uses ``StreamFilterComponent``
to manipulate the data such as applying transformations and running AI inference.

Let us walk through a simple example such `monaistream-pytorch-pp-app <LINKREF_GITHUB_MONAISTREAM/sample/monaistream-pytorch-pp-app/main.py>`_
where the pipeline can be visualized as shown below.

.. mermaid::

  stateDiagram-v2
      NVAggregatedSourcesBin --> NVVideoConvert: BatchData Output
      NVVideoConvert --> NVInferServer: RGBA Output
      NVInferServer --> TransformChainComponent: Original Image
      NVInferServer --> TransformChainComponent: User Metadata
      TransformChainComponent --> NVEglGlesSink: <center>Selected Output Key<br>in TransformChainComponent</center>


There are five important components in this pipeline that will apply to most of the use cases.


MONAI Stream Pipeline components
================================

URISource
---------

``URISource`` is a source component which generates data when provided with a URI to a data source such as a local video file,
a video stream (e.g. via ``http://``), or a live stream.

.. note::
   
   Feed from multiple ``URISource`` components can be aggregated together via ``NVAggregatedSourcesBin`` (see below).

AJAVideoSource
--------------

``AJAVideoSource`` provides support for live streaming from AJA capture devices connected to `x86` or `Clara AGX` systems.

.. warning::
   
   ``AJAVideoSource`` is currently not compatible with ``NVAggregatedSourcesBin``.

NVAggregatedSourcesBin
----------------------

``NVAggregatedSourcesBin`` is a special type of `source` component, which can aggregate data from multiple sources
and present it as a single unit of data concatenated in the batch dimension. For instance it may concatenate multiple
``URISource`` components as shown below, resizing all to the same dimensions, and stacking the data in the batch dimension.

.. mermaid::

   stateDiagram-v2
      state "URISource" as URISource_1
      state "URISource" as URISource_2
      state NVAggregatedSourcesBin {
         URISource_1 --> BatchData
         URISource_2 --> BatchData
         BatchData --> [*]
      }

NVVideoConvert
--------------

``NVVideoConvert`` then processes the output of ``NVAggregatedSourcesBin`` to convert it to ``RGBA``.

NVInferServer
-------------

``NVInferServer`` receives the output of ``NVVideoConvert`` and runs a configured AI model to produce results (e.g. segmentation, classification, etc.)
in the form of `User Metadata`. This means that ``NVInferServer`` outputs primarily the original input along with inference results in user medatadata,
therefore one must be careful to select the correct data in the following component.

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

TransformChainComponent
-----------------------

``TransformChainComponent`` is a filter component which allows the developer to apply MONAI transformations to streaming data coming from
any other MONAI Stream `source` or `filter`. In the pipeline shown above ``TransformChainComponent`` takes all outputs from ``NVInferServer``,
namely the original stream and the segmentation output, and combines them together to show the segmentation overlaid on the original video stream.

.. mermaid::

   stateDiagram-v2
      state TransformChainComponent {
         [*] --> Activationsd
         Activationsd --> AsDiscreted
         AsDiscreted --> AddChanneld
         AddChanneld --> AsChannelLastd
         AsChannelLastd --> ConcatItemsd
         ConcatItemsd --> Lambdad
         Lambdad --> CastToTyped
         CastToTyped --> [*]
      }

NVEglGlesSink
-------------

``NVEglGlesSink`` is a component that allows developers to visualize the outputs of their pipelines when data is streamed via NVIDIA GPU.

FakeSink
--------

``FakeSink`` is a sink component that allows the developer to end the MONAI Stream pipeline without the need to visualize data. ``FakeSink``
is useful for unit testing and for cases where ``TransformChainComponent`` outputs data to disk, but provides no output other than the original
data stream.
