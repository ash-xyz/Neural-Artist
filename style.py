import argparse
import os
import sys

from src.train import train
from src.eval import stylize
from src.cam import webcam
from src.video import transform_video
from src.export import export

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    subparser = main_parser.add_subparsers(
        title="subcommands", dest="subcommand")

    train_arg_parser = subparser.add_parser(
        "train", help="parser for training arguments")

    train_arg_parser.add_argument('--dataset', type=str, required=True,
                                  help='Path to dataset')
    train_arg_parser.add_argument('--style', type=str, required=True,
                                  help='Path to the style image')
    train_arg_parser.add_argument('--image-size', type=int, default=256,
                                  help='Size of training images')
    train_arg_parser.add_argument('--content-weight', type=float, default=1,
                                  help='Weight for content loss')
    train_arg_parser.add_argument('--style-weight', type=float, default=2e2,
                                  help='Weight for style loss')
    train_arg_parser.add_argument('--tv-weight', type=float, default=1e-6,
                                  help='Weight for tv loss')
    train_arg_parser.add_argument('--norm-range', type=bool, default=True,
                                  help='Normalization Range for VGG input, True for 0,255 False for 0,1')

    train_arg_parser.add_argument('--epochs', type=int, default=2)
    train_arg_parser.add_argument('--batch-size', type=int, default=4)
    train_arg_parser.add_argument('--learning-rate', type=float, default=1e-3)
    train_arg_parser.add_argument('--log-interval', type=int, default=500)
    train_arg_parser.add_argument('--save-interval', type=int, default=2000)
    train_arg_parser.add_argument(
        '--checkpoint-dir', type=str, default='models')

    eval_arg_parser = subparser.add_parser(
        "eval", help="parser for stylizing arguments")
    eval_arg_parser.add_argument('--content', type=str, required=True,
                                 help='Path to the content image')
    eval_arg_parser.add_argument('--out', type=str, required=True,
                                 help='Path to the result image')
    eval_arg_parser.add_argument('--model', type=str, required=True,
                                 help='Path to the style model')

    cam_arg_parser = subparser.add_parser("cam", help="parser for webcam")
    cam_arg_parser.add_argument('--model', type=str, required=True,
                                help="Path to style model")
    cam_arg_parser.add_argument('--width', type=int, default=None)
    cam_arg_parser.add_argument('--height', type=int, default=None)

    vid_arg_parser = subparser.add_parser("video", help="parser for videos")
    vid_arg_parser.add_argument('--model', type=str, required=True,
                                help="Path to style model")
    vid_arg_parser.add_argument('--output-dir', type=str, required=True,
                                help='Path to the result video')
    vid_arg_parser.add_argument('--content', type=str, required=True,
                                help='Path to the content video')
    vid_arg_parser.add_argument('--show-frame', type=bool, default=False,
                                help='Shows current frame being processed')

    export_arg_parser = subparser.add_parser(
        "export", help="parser for onnx exports")
    export_arg_parser.add_argument('--model', type=str, required=True,
                                   help="Path to style model")
    export_arg_parser.add_argument('--output-dir', type=str, default='models',
                                   help="output path")
    args = main_parser.parse_args()

    if args.subcommand is None:
        print("Specify either train, eval, video or cam")
        sys.exit(1)
    if args.subcommand == "train":
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        train(args)
    if args.subcommand == "eval":
        stylize(args)
    if args.subcommand == "cam":
        webcam(args)
    if args.subcommand == "video":
        transform_video(args)
    if args.subcommand == "export":
        export(args)
