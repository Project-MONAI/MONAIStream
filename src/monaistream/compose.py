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
from typing import Sequence

from gi.repository import GLib, Gst

from monaistream.errors import BinCreationError, StreamComposeCreationError, StreamTransformChainError
from monaistream.filters.convert import NVVideoConvert
from monaistream.filters.infer import NVInferServer
from monaistream.interface import (
    AggregatedSourcesComponent,
    InferenceFilterComponent,
    StreamComponent,
    StreamSourceComponent,
)
from monaistream.sources.ajavideosrc import AJAVideoSource

logger = logging.getLogger(__name__)


class StreamCompose(object):
    """
    MONAI Stream pipeline composer is the core function that allows MONAI Stream and MONAI core elements to integrate.
    """

    def __init__(self, components: Sequence[StreamComponent]):
        """
        At initialization all components in the pipeline are initilized thought the `initialize` method, and are then
        linked by retrieving their underlying GStreamer elements through `get_gst_element`.

        :param components: is a sequence of `StreamComponent` from which all components in MONAI Stream SDK are inherited
        """
        self._pipeline = Gst.Pipeline()
        self._exception = None

        # initialize and configure components
        # link the sources and sinks between the aggregator and multiplexer
        # configure batch size in nvinfer server
        batch_size = 1
        src_is_live = False
        insert_muxer = any([isinstance(c, NVInferServer) for c in components])
        for component in components:
            component.initialize()

            for elem in component.get_gst_element():
                self._pipeline.add(elem)

            if isinstance(component, StreamSourceComponent):
                src_is_live = component.is_live()

            insert_muxer = insert_muxer and (
                not isinstance(component, AggregatedSourcesComponent) and not isinstance(component, AJAVideoSource)
            )

            # set the batch size of nvinferserver if it exists in the pipeline
            # from the number of sources otherwise assume there's only one source
            if isinstance(component, AggregatedSourcesComponent):
                batch_size = component.get_num_sources()
            elif isinstance(component, InferenceFilterComponent):
                component.set_batch_size(batch_size)

        # link the components in the chain
        for idx in range(len(components) - 1):

            curr_component = components[idx]
            curr_component_elems = curr_component.get_gst_element()
            curr_component_elem = curr_component_elems[-1]
            next_component_elem = components[idx + 1].get_gst_element()[0]

            # link subelements of element (e.g. converters and capsfilters in NVVideoConvert components)
            for subidx in range(len(curr_component_elems) - 1):

                # an aggregated source is a special component that contains a muxer which
                # is necessary to batch data from all the sources listed in the aggregator
                if isinstance(components[idx], AggregatedSourcesComponent):
                    source, muxer = curr_component_elems
                    num_sources = components[idx].get_num_sources()

                    for src_idx in range(num_sources):

                        # get a sinkpad for each source in the stream multiplexer
                        sinkpad = muxer.get_request_pad(f"sink_{src_idx}")
                        if not sinkpad:
                            raise StreamComposeCreationError(
                                f"Unable to create multiplexer sink pad bin for {component.get_name()}"
                            )

                        # get the source pad from the upstream component
                        srcpad = source.get_static_pad("src")
                        if not srcpad:
                            raise StreamComposeCreationError(f"Unable to create bin src pad for {component.get_name()}")

                        link_code = srcpad.link(sinkpad)
                        if link_code != Gst.PadLinkReturn.OK:
                            logger.error(
                                f"Linking of source and multiplexer for component {component.get_name()}"
                                f" failed: {link_code.value_nick}"
                            )
                            exit(1)

                # other components are assumed to not need pad information to be able to link the Gst elements
                # container within the component, unless there is a need to insert a muxer
                else:

                    link_code = curr_component_elems[subidx].link(curr_component_elems[subidx + 1])
                    if not link_code:
                        logger.error(f"Creation of {components[idx].get_name()} failed")
                        exit(1)

            if isinstance(curr_component, NVVideoConvert) and insert_muxer:
                # a multiplexer is necessary when `nvinferserver`` is present as it provides batch
                # metadata to the pipeline which nvinferserver can consume
                muxer = Gst.ElementFactory.make("nvstreammux", f"{curr_component.get_name()}-nvstreammux")
                if not muxer:
                    raise BinCreationError(
                        f"Unable to create multiplexer for {curr_component.__class__._name}"
                        f" with name {curr_component.get_name()}"
                    )

                muxer.set_property("batch-size", batch_size)

                src_prop_names = [c.name for c in curr_component_elem.list_properties()]
                if (
                    "caps" in src_prop_names
                    and curr_component_elem.get_property("caps").get_structure(0).get_int("width")[0]
                ):
                    muxer.set_property(
                        "width", curr_component_elem.get_property("caps").get_structure(0).get_int("width").value
                    )

                if (
                    "caps" in src_prop_names
                    and curr_component_elem.get_property("caps").get_structure(0).get_int("height")[0]
                ):
                    muxer.set_property(
                        "height", curr_component_elem.get_property("caps").get_structure(0).get_int("height").value
                    )

                muxer.set_property("live-source", src_is_live)

                # get a sinkpad from the multiplexer
                sinkpad = muxer.get_request_pad("sink_0")
                if not sinkpad:
                    raise StreamComposeCreationError(
                        f"Unable to create multiplexer sink pad bin for {component.get_name()}"
                    )

                # get the source pad from the current source
                srcpad = curr_component_elem.get_static_pad("src")
                if not srcpad:
                    raise StreamComposeCreationError(f"Unable to create bin src pad for {component.get_name()}")

                link_code = srcpad.link(sinkpad)
                if link_code != Gst.PadLinkReturn.OK:
                    logger.error(
                        f"Linking of source and multiplexer for component {component.get_name()}"
                        f" failed: {link_code.value_nick}"
                    )
                    exit(1)

                link_code = muxer.link(next_component_elem)

                if not link_code:
                    logger.error(
                        f"Linking of {components[idx].get_name()}-multiplexer and "
                        f"{components[idx + 1].get_name()} failed"
                    )
                    exit(1)

            else:

                link_code = curr_component_elem.link(next_component_elem)

                if not link_code:
                    logger.error(
                        f"Linking of {components[idx].get_name()} and " f"{components[idx + 1].get_name()} failed"
                    )
                    exit(1)

    def bus_call(self, bus, message, loop):
        if message.type == Gst.MessageType.EOS:
            logger.info("[INFO] End of stream")
            loop.quit()

        elif message.type == Gst.MessageType.INFO:
            info, debug = message.parse_info()
            logger.info("[INFO] {}: {}".format(info, debug))

        elif message.type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warn("[WARN] {}: {}".format(err, debug))

        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("[EROR] {}: {}".format(err, debug))
            loop.quit()
            self._exception = StreamTransformChainError(f"Pipeline failed - {err}: {debug}")

        elif message.type == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            logger.debug("State changed from %s to %s (pending=%s)", old.value_name, new.value_name, pending.value_name)
            Gst.debug_bin_to_dot_file(
                self._pipeline, Gst.DebugGraphDetails.ALL, f"{self._pipeline.name}-{old.value_name}-{new.value_name}"
            )

        elif message.type == Gst.MessageType.STREAM_STATUS:
            type_, owner = message.parse_stream_status()
            logger.debug("Stream status changed to %s (owner=%s)", type_.value_name, owner.name)
            Gst.debug_bin_to_dot_file(
                self._pipeline, Gst.DebugGraphDetails.ALL, f"{self._pipeline.name}-{type_.value_name}"
            )

        elif message.type == Gst.MessageType.DURATION_CHANGED:
            logger.debug("Duration changed")

        return True

    def __call__(self) -> None:
        loop = GLib.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()

        bus.connect("message", self.bus_call, loop)

        self._pipeline.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        finally:
            if self._exception:
                raise self._exception
            self._pipeline.set_state(Gst.State.NULL)
