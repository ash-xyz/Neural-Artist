"""Exports trained models to be used as .onnx files"""
import os
import torch
from src.transformer_net import TransformerNet


def export(args):
    """Exports pytorch model to be used as .onnx """
    # Loads Pytorch Model
    transformer = TransformerNet()
    state_dict = torch.load(args.model)
    transformer.load_state_dict(state_dict)

    # Gets save_path name
    model_name = args.model.split('/')[-1].split('.')[0]
    save_path = os.path.join(args.output_dir, '{}.onnx'.format(model_name))

    # Exports ONNX Model
    x = torch.randn(1, 3, 1920, 1080)
    torch.onnx.export(transformer, x, save_path, opset_version=11)
