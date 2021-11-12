import argparse
import logging
import os
import pathlib
from typing import Any, List

from monaistream.util.convert import to_onnx, to_trt

CMD_ACTIONS = ["convert"]


class Entry:
    def __init__(
        self,
        loglevel: Any = logging.ERROR,
        actions: List[str] = CMD_ACTIONS,
        logformat: str = (
            "[%(asctime)s] [%(process)s] [%(threadName)s] " "[%(levelname)s] (%(name)s:%(lineno)d) - %(message)s"
        ),
    ) -> None:
        self.actions = actions
        logging.basicConfig(level=loglevel, format=logformat)

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("MONAI Stream command line utility")
        subparsers = parser.add_subparsers(help="sub-command help")

        if CMD_ACTIONS[0] in self.actions:
            conv_parser = subparsers.add_parser("convert", help="Model converter utility")
            conv_parser.add_argument("-i", "--input_model", required=True, help="The filename of the input model")
            conv_parser.add_argument("-o", "--output_model", required=True, help="The filename of the output model")
            conv_parser.add_argument(
                "-I", "--model_inputs", required=True, nargs="+", help="A list of model input names"
            )
            conv_parser.add_argument(
                "-O", "--model_outputs", required=True, nargs="+", help="A list of model output names"
            )
            conv_parser.add_argument(
                "-S",
                "--input_size",
                required=True,
                type=int,
                nargs="+",
                action="append",
                help=(
                    "The shapes of the inputs to the model in the "
                    "same order specified in the `--model_inputs` argument"
                ),
            )
            conv_parser.add_argument("-w", "--workspace", type=int, default=1000)
            conv_parser.set_defaults(action="convert")

        return parser

    def action_convert(self, args):
        if len(args.model_inputs) != len(args.input_size):
            print("The number of model input names must match the number of model input sizes")
            exit(1)

        if not (args.input_model.endswith(".pt") or args.input_model.endswith(".ts")):
            print(f"Input model must be PyTorch (.pt) or TorchScript (.ts): {args.input_model}")
            exit(1)

        if not (args.output_model.endswith(".onnx") or args.output_model.endswith(".engine")):
            print(f"Output model must be ONNX (.onnx) or TRT (.engine): {args.output_model}")
            exit(1)

        if args.input_model.endswith(".onnx"):
            to_onnx(
                input_model_path=args.input_model,
                output_model_path=args.output_model,
                input_names=args.model_inputs,
                output_names=args.model_outputs,
                input_sizes=args.input_size,
                do_constant_folding=False,
            )
        else:
            tmp_onnx_file = args.output_model.replace(pathlib.Path(args.output_model).suffix, "") + ".onnx"
            to_onnx(
                input_model_path=args.input_model,
                output_model_path=tmp_onnx_file,
                input_names=args.model_inputs,
                output_names=args.model_outputs,
                input_sizes=args.input_size,
                do_constant_folding=True,
            )
            try:
                to_trt(
                    input_model_path=tmp_onnx_file,
                    output_model_path=args.output_model,
                    workspace=args.workspace,
                )
            finally:
                os.remove(tmp_onnx_file)

    def run(self):
        parser = self.create_parser()
        args = parser.parse_args()

        if not hasattr(args, "action"):
            parser.print_usage()
            exit(-1)

        if args.action == CMD_ACTIONS[0]:
            self.action_convert(args)
        else:
            parser.print_help()
            exit(-1)


def main():
    Entry().run()


if __name__ == "__main__":
    main()
