import logging
from typing import Any, Sequence, Tuple, Union

from gi.repository import GObject, Gst

from monaistream.errors import StreamComposeCreationError
from monaistream.interface import (
    AggregatedSourcesComponent,
    InferenceFilterComponent,
    StreamComponent,
    StreamSourceComponent,
)

logger = logging.getLogger(__name__)


class StreamCompose(object):
    def __init__(self, components: Sequence[StreamComponent]) -> None:
        self._pipeline = Gst.Pipeline()

        # initialize and configure components
        # link the sources and sinks between the aggregator and multiplexer
        # configure batch size in nvinfer server
        batch_size = 1
        for component in components:
            component.initialize()

            # add elements from stream components to pipeline
            if isinstance(component.get_gst_element(), tuple):
                for elem in component.get_gst_element():
                    self._pipeline.add(elem)
            else:
                self._pipeline.add(component.get_gst_element())

            # set the batch size of nvinferserver if it exists in the pipeline
            # from the number of sources
            if isinstance(component, AggregatedSourcesComponent):
                batch_size = component.get_num_sources()
            elif isinstance(component, InferenceFilterComponent):
                component.set_batch_size(batch_size)

        # link the components in the chain
        for idx in range(len(components) - 1):

            elems: Union[Any, Tuple[Any]] = ()

            if isinstance(components[idx].get_gst_element(), tuple):
                elems = components[idx].get_gst_element()
                connect_component_prev = elems[-1]
            else:
                connect_component_prev = components[idx].get_gst_element()

            if isinstance(components[idx + 1].get_gst_element(), tuple):
                connect_component_next = components[idx + 1].get_gst_element()[0]
            else:
                connect_component_next = components[idx + 1].get_gst_element()

            # link subelements of element (e.g. converters and capsfilters in NVVideoConvert components)
            for subidx in range(len(elems) - 1):

                if isinstance(components[idx], StreamSourceComponent):
                    source, muxer = elems

                    num_sources = 1
                    if isinstance(components[idx], AggregatedSourcesComponent):
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

                else:
                    link_code = elems[subidx].link(elems[subidx + 1])
                    if not link_code:
                        logger.error(f"Creation of {components[idx].get_name()} failed")
                        exit(1)

            link_code = connect_component_prev.link(connect_component_next)

            if not link_code:
                logger.error(f"Linking of {components[idx].get_name()} and " f"{components[idx + 1].get_name()} failed")
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
        loop = GObject.MainLoop()
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()

        bus.connect("message", self.bus_call, loop)

        self._pipeline.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        except Exception:
            pass

        self._pipeline.set_state(Gst.State.NULL)
