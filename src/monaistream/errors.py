class BinCreationError(Exception):
    pass


class StreamComposeCreationError(Exception):
    pass


class StreamComposeCreationStructureError(StreamComposeCreationError):
    pass


class StreamProbeCreationError(Exception):
    pass


class StreamProbeRuntimeError(Exception):
    pass


class StreamTransformChainError(Exception):
    pass


class StreamTransormChainNoRegisteredCallbackError(StreamTransformChainError):
    pass
