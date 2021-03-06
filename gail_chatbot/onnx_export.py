"""Console script for exporting chatbots to onnx"""
import sys
import click
from transformers import BartForConditionalGeneration, AutoTokenizer
import torch
from torch import onnx


@click.group()
def main(args=None):

    return 0


class BartWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        return self.model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True
        ).logits


@main.command()
@click.option(
    "-m", "--model-path", type=click.STRING, help="Path to the pretrained bart model"
)
@click.option(
    "-t",
    "--tokenizer-path",
    type=click.STRING,
    help="Path to the pretrained bart model",
    default=None,
)
def bart(model_path, tokenizer_path):
    bart = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path is not None else model_path
    )
    example_input_ids = tokenizer("Example ctx", return_tensors="pt").input_ids
    other = example_input_ids.clone()
    bart_wrapper = BartWrapper(bart)
    onnx.export(
        bart_wrapper,
        (example_input_ids, other),
        "bart.onnx",
        input_names=["input_ids", "decoder_input_ids"],
        output_names=["logits"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            "input_ids": {1: "text_seq"},
            "decoder_input_ids": {1: "text_seq"},
            "logits": {1: "logits_seq"},
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
