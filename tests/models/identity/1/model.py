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

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):

        responses = []

        for request in requests:

            # get the input by name (as configured in config.pbtxt)
            input_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()

            output_0 = np.copy(input_0)

            output0_tensor = pb_utils.Tensor("OUTPUT0", output_0.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output0_tensor],
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
