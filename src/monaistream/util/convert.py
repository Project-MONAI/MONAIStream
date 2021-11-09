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

import os
import pathlib
import subprocess
import sys
from typing import List

import torch.onnx


def to_onnx(
    input_model_path: str,
    output_model_path: str,
    input_names: List[str],
    output_names: List[str],
    input_sizes: List[List[int]],
    do_constant_folding: bool = False,
) -> None:

    model_inputs = []
    for input_size in input_sizes:
        model_inputs.append(torch.randn(*input_size))

    torch_model = torch.jit.load(input_model_path)
    torch_model = torch_model.eval()

    torch.onnx.export(
        torch_model,
        model_inputs,
        output_model_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=do_constant_folding,
    )


def to_trt(
    input_model_path: str,
    output_model_path: str,
    explicit_batch: bool = True,
    verbose: bool = False,
    workspace: int = 1000,
) -> None:

    subprocess.check_call([sys.executable, "-m", "pip", "install", "nvidia-pyindex"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx-graphsurgeon"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "polygraphy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])

    sfx = pathlib.Path(input_model_path).suffix
    folded_model_path = input_model_path.replace(sfx, f"_folded{sfx}")
    fold_command = [
        "polygraphy",
        "surgeon",
        "sanitize",
        f"{input_model_path}",
        "--fold-constants",
        f"--output={folded_model_path}",
    ]
    convert_command = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={folded_model_path}",
        f"--saveEngine={output_model_path}",
    ]

    if explicit_batch:
        convert_command.append("--explicitBatch")

    if verbose:
        convert_command.append("--verbose")

    if workspace <= 0:
        raise ValueError("Invalid `workspace` value provided for TRT model conversion")

    convert_command.append(f"--workspace={workspace}")

    print(" ".join(fold_command))
    subprocess.run(fold_command)
    subprocess.run(convert_command)
    os.remove(folded_model_path)
