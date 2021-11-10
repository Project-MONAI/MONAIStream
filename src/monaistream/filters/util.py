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

from typing import Any

import numpy as np
import pyds


def get_nvdstype_size(nvds_type: pyds.NvDsInferDataType) -> int:

    if nvds_type == pyds.NvDsInferDataType.INT8:
        return 1
    elif nvds_type == pyds.NvDsInferDataType.HALF:
        return 2
    elif nvds_type == pyds.NvDsInferDataType.INT32:
        return 4
    elif nvds_type == pyds.NvDsInferDataType.FLOAT:
        return 4

    return 4


def get_nvdstype_npsize(nvds_type: pyds.NvDsInferDataType) -> Any:

    if nvds_type == pyds.NvDsInferDataType.INT8:
        return np.int8
    elif nvds_type == pyds.NvDsInferDataType.HALF:
        return np.half
    elif nvds_type == pyds.NvDsInferDataType.INT32:
        return np.int32
    elif nvds_type == pyds.NvDsInferDataType.FLOAT:
        return np.float32

    return np.float32
