"""Exports trained models to be used as .onnx files"""
import os
import torch
from src.transformer_net import TransformerNet


def export(args):
    """Exports pytorch model to be used as .onnx """
    # Loads Pytorch Model
    transformer = TransformerNet()
    transformer.load_state_dict(torch.load(args.model))
    transformer.eval()

    # Gets save_path name
    model_name = args.model.split('/')[-1].split('.')[0]
    save_path = os.path.join(args.output_dir, '{}.onnx'.format(model_name))

    # Exports ONNX Model
    x = torch.randn(1, 3, 'N', 'N')
    torch.onnx.export(transformer, x, save_path, opset_version=11)
