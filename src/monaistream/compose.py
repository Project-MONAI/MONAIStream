import logging
from typing import Any, Sequence, Tuple, Union

from gi.repository import GObject, Gst

from monaistream.errors import StreamComposeCreationError
from monaistream.interface import AggregatedSourcesComponent, InferenceFilterComponent, StreamComponent

logger = logging.getLogger(__name__)


class StreamCompose(object):
    def __init__(self, components: Sequence[StreamComponent]) -> None:
        self._pipeline = Gst.Pipeline()

        # initialize and configure components
        # link the sources and sinks between the aggregator and multiplexer
        # configure batch size in nvinfer server
        first_filter_index = -1
        batch_size = 1
        for component_idx, component in enumerate(components):
            component.initialize()

            # some elements return tuples (e.g. `NVVideoConvert`)
            if isinstance(component.get_gst_element(), tuple):
                for elem in component.get_gst_element():
                    self._pipeline.add(elem)
            else:
                self._pipeline.add(component.get_gst_element())

            if isinstance(component, AggregatedSourcesComponent):

                batch_size = component.get_num_sources()

                first_filter_index = component_idx + 1

                # each source in the aggregator pad (assumed to be the component before
                # the multiplexer) will be linked to a sink of the multiplexer pad
                source_bin, muxer = component.get_gst_element()

                for src_idx in range(component.get_num_sources()):

                    # get a sinkpad for each source in the stream multiplexer
                    sinkpad = muxer.get_request_pad(f"sink_{src_idx}")
                    if not sinkpad:
                        raise StreamComposeCreationError(
                            f"Unable to create multiplexer sink pad bin for {component.get_name()}"
                        )

                    # get the source pad from the upstream component
                    srcpad = source_bin.get_static_pad("src")
                    if not srcpad:
                        raise StreamComposeCreationError(f"Unable to create bin src pad for {component.get_name()}")

                    link_code = srcpad.link(sinkpad)
                    if link_code != Gst.PadLinkReturn.OK:
                        logger.error(
                            f"Linking of source bin and multiplexer for component {component.get_name()}"
                            f" failed: {link_code.value_nick}"
                        )
                        exit(1)

            elif isinstance(component, InferenceFilterComponent):

                component.set_batch_size(batch_size)

        # link the components in the chain
        for idx in range(first_filter_index, len(components)):

            elems: Union[Any, Tuple[Any]] = ()

            if isinstance(components[idx - 1].get_gst_element(), tuple):
                connect_component_prev = components[idx - 1].get_gst_element()[-1]
            else:
                connect_component_prev = components[idx - 1].get_gst_element()

            if isinstance(components[idx].get_gst_element(), tuple):
                elems = components[idx].get_gst_element()
                connect_component_next = elems[0]
            else:
                connect_component_next = components[idx].get_gst_element()

            # link subelements of element (e.g. converters and capsfilters in NVVideoConvert components)
            for subidx in range(len(elems) - 1):
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
