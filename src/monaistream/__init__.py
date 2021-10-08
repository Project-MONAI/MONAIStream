import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GstAudio", "1.0")

from gi.repository import Gst

Gst.init(None)

from . import _version

version_dict = _version.get_versions()

__version__ = version_dict.get("version", "0+unknown")
__revision_id__ = version_dict.get("full-revisionid")
__copyright__ = "(c) 2021 MONAI Consortium"

del version_dict


def print_config(file=sys.stdout):

    from collections import OrderedDict

    import numpy as np
    import torch

    output = OrderedDict()
    output["MONAIStream"] = __version__
    output["Numpy"] = np.version.full_version
    output["Pytorch"] = torch.__version__

    print(__copyright__)

    for k, v in output.items():
        print(f"{k} version: {v}", file=file, flush=True)

    print(f"MONAILabel rev id: {__revision_id__}")
