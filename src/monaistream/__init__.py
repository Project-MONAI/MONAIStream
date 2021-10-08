import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GstAudio", "1.0")

from gi.repository import GObject, Gst

Gst.init(None)
GObject.threads_init()

from . import _version
__version__ = _version.get_versions()['version']
