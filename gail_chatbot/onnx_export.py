"""Console script for exporting chatbots to onnx"""
import sys
import click
from transformers import BartForConditionalGeneration, AutoTokenizer
import torch
from torch import onnx


@click.group()
def main(args=None):

    return 0


class BartDecoderWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, encoder_hidden_state, decoder_input_ids):
        output = self.model(
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            output_hidden_states=True,
            encoder_outputs=(encoder_hidden_state, None, None),
        )
        return output.logits, output.encoder_last_hidden_state


class BartEncoderWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        output = self.model.model.encoder(input_ids=input_ids)
        return output[0]


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
@click.option(
    "-o",
    "--output-path",
    type=click.STRING,
    help="Path to the write model to",
    default="bart.onnx",
)
def bart(model_path, tokenizer_path, output_path):
    bart = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path is not None else model_path
    )
    example_input_ids = tokenizer("Example ctx", return_tensors="pt").input_ids
    other = example_input_ids.clone()
    encoder_output = torch.empty([1, 0, 1024], dtype=torch.float32)
    bart_enc_wrapper = BartEncoderWrapper(bart)
    bart_dec_wrapper = BartDecoderWrapper(bart)

    onnx.export(
        bart_enc_wrapper,
        (example_input_ids,),
        "encoder_" + output_path,
        input_names=["input_ids"],
        output_names=["encoder_hidden_state_out"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            "input_ids": {1: "text_seq"},
            "encoder_hidden_state_out": {1: "text_seq"},
        },
    )
    onnx.export(
        bart_dec_wrapper,
        (encoder_output, other),
        "decoder_" + output_path,
        input_names=["encoder_hidden_state", "decoder_input_ids"],
        output_names=["logits"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            "encoder_hidden_state": {1: "enc_cahce_seq"},
            "decoder_input_ids": {1: "text_seq_inputs"},
            "logits": {1: "logits_seq"},
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
