#!/usr/bin/env python3

################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import sys

sys.path.append("../")
import ctypes
import math
import os
import os.path
from ctypes import *

import cupy
import cupy.cuda.cudnn
import cupy.cudnn
import cv2
import gi
import numpy as np
from common.bus_call import bus_call
from common.FPS import GETFPS
from common.is_aarch_64 import is_aarch64
from gi.repository import GObject, Gst

import pyds

gi.require_version("Gst", "1.0")


fps_streams = {}


VIDEO_WIDTH = 1264
VIDEO_HEIGHT = 1024

# 1 / frames_per_second * 10^6
MUXER_BATCH_TIMEOUT_USEC = 16666

GST_CAPS_FEATURES_NVMM = "memory:NVMM"


def q_src_pad_buffer_probe(pad, info, u_data):

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number = frame_meta.frame_num
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            if debug_mode:
                print("There is user meta data at queue2 at frame {}".format(frame_number))
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                continue

            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

            # this should get us only "OUTPUT__0" as the monai unet has only one output
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

            # start to do things in cupy
            # Create dummy owner object to keep memory for the image array alive
            owner = None
            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
            data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(
                hash(gst_buffer), frame_meta.batch_id
            )
            # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

            # Create UnownedMemory object from the gpu buffer
            unownedmem = cupy.cuda.UnownedMemory(ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None), size, owner)
            # Create MemoryPointer object from unownedmem, at index 0
            memptr = cupy.cuda.MemoryPointer(unownedmem, 0)
            # Create cupy array to access the image data. This array is in GPU buffer
            n_frame_gpu = cupy.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order="C")
            # n_frame_gpu values range from 0 - 255

            """If we want to convert NvDsInferLayerInfo buffer to numpy array and use opencv,
            # from this forum https://forums.developer.nvidia.com/t/urgent-how-to-convert-deepstream-tensor-to-numpy/128601/9
            # access the seg mask as follows
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            v = np.ctypeslib.as_array(ptr, shape=(VIDEO_HEIGHT, VIDEO_WIDTH))
            """

            # create UnownedMemory object from the segmentation mask gpu buffer
            seg_unownedmem = cupy.cuda.UnownedMemory(
                ctypes.pythonapi.PyCapsule_GetPointer(layer.buffer, None),
                ctypes.sizeof(ctypes.c_float) * VIDEO_WIDTH * VIDEO_HEIGHT,
                owner,
            )
            # Create MemoryPointer object from unownedmem, at index 0
            seg_memptr = cupy.cuda.MemoryPointer(seg_unownedmem, 0)
            # Create cupy array to access the seg mask data. This array is in GPU buffer
            seg_mask_gpu = cupy.ndarray(shape=(VIDEO_HEIGHT, VIDEO_WIDTH), dtype=ctypes.c_float, memptr=seg_memptr)

            # debug only
            if debug_mode:
                # Use cp.asnumpy to move array into CPU buffer to perform cv2 operations
                mask_cpu = cupy.asnumpy(seg_mask_gpu)
                # convert python array into numpy array format in the copy mode.
                mask_numpy = np.array(mask_cpu, copy=True, order="C")
                # convert the array into cv2 default color format
                mask_numpy = cv2.cvtColor(mask_numpy, cv2.COLOR_RGBA2BGRA)
                img_path = "{}/stream_{}/rawMask_frame_{}.jpg".format(folder_name, frame_meta.pad_index, frame_number)
                cv2.imwrite(img_path, mask_cpu)
                print("Raw mask min val: {}".format(mask_cpu.min()))
                print("Raw mask max val: {}".format(mask_cpu.max()))
            # end debug

            stream = cupy.cuda.stream.Stream()
            stream.use()
            # seg_mask_activated is of range [0,1]
            seg_mask_activated = cupy.cudnn.activation_forward(seg_mask_gpu, cupy.cuda.cudnn.CUDNN_ACTIVATION_SIGMOID)
            # modify only the red channel to show mask
            n_frame_gpu[:, :, 0] = cupy.multiply(1.0 - seg_mask_activated, n_frame_gpu[:, :, 0])
            stream.synchronize()

            try:
                l_user = l_user.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    if debug_mode:
        print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    if debug_mode:
        print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if name.find("nvv4l2decoder") != -1:
        Object.set_property("num-extra-surfaces", 4)
        Object.set_property("cudadec-memtype", 0)


def create_source_bin(index, uri):
    if debug_mode:
        print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    if debug_mode:
        print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN] <mode: debug or performance> \n" % args[0])
        sys.exit(1)

    for i in range(0, len(args) - 2):
        fps_streams["stream{0}".format(i)] = GETFPS(i)
    number_sources = len(args) - 2

    global debug_mode
    debug_mode = False
    if args[-1] == "debug":
        debug_mode = True
    elif args[-1] == "performance":
        debug_mode = False
    else:
        sys.stderr.write("Must specify debug or performance in the command line, defaulting to performance \n")

    global folder_name
    folder_name = "output_frames"

    if debug_mode:
        os.makedirs(folder_name, exist_ok=True)
        print("Output seg masks will be saved in ", folder_name)
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    if debug_mode:
        print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    if debug_mode:
        print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        if debug_mode:
            os.makedirs(folder_name + "/stream_" + str(i), exist_ok=True)
            print("Creating source_bin ", i, " \n ")
        uri_name = args[i + 1]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.

    if debug_mode:
        print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")

    if debug_mode:
        print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    if is_aarch64():
        if debug_mode:
            print("Creating transform \n ")
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    # Running pgie on separate thread to not block buffers on main thread
    if debug_mode:
        print("Creating queue1 \n ")
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    if not queue1:
        sys.stderr.write(" Unable to create queue1 \n")

    if debug_mode:
        print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Running CuPy post processing on separate thread to not block buffers
    # on main thread
    if debug_mode:
        print("Creating queue2 \n ")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    if not queue2:
        sys.stderr.write(" Unable to create queue2 \n")

    if debug_mode:
        print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        if debug_mode:
            print("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)

    streammux.set_property("width", VIDEO_WIDTH)
    streammux.set_property("height", VIDEO_HEIGHT)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("buffer-pool-size", 4)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("nvbuf-memory-type", 2)

    nvvidconv1.set_property("nvbuf-memory-type", 2)

    pgie.set_property("config-file-path", "configs/config_unet_pytorch_nopostprocess.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    if debug_mode:
        print("Adding elements to Pipeline \n")
    pipeline.add(nvvidconv1)
    pipeline.add(filter1)
    pipeline.add(queue1)
    pipeline.add(pgie)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(queue2)
    pipeline.add(sink)

    if debug_mode:
        print("Linking elements in the Pipeline \n")
    streammux.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    if is_aarch64():
        queue2.link(transform)
        transform.link(sink)
    else:
        queue2.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Add a probe on the primary-infer source pad to get inference output tensors
    q_src_pad = queue2.get_static_pad("src")
    if not q_src_pad:
        sys.stderr.write(" Unable to get src pad of queue2 \n")
    else:
        q_src_pad.add_probe(Gst.PadProbeType.BUFFER, q_src_pad_buffer_probe, 0)

    # List the sources
    if debug_mode:
        print("Now playing...")
        for i, source in enumerate(args[:-1]):
            if i != 0:
                print(i, ": ", source)

    if debug_mode:
        print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
