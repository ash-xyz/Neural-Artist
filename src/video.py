import torch
from torchvision import transforms
from src.transformer_net import TransformerNet
import cv2
from tqdm import tqdm
import numpy as np


def transform_video(args):
    """Stylizes videos
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Model
    transformer = TransformerNet()
    state_dict = torch.load(args.model)
    transformer.load_state_dict(state_dict)
    transformer.to(device)

    # Image Transforms Preprocessing
    preprocess = transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor(),
         transforms.Lambda(lambda x: x.mul(255))])

    # OpenCV Video Capture Info
    video = cv2.VideoCapture(args.content)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Stylizing Frame
    print("Stylizing Frames:")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vout = cv2.VideoWriter("out.mp4", fourcc, fps, (width, height))
    with torch.no_grad():
        for i in tqdm(range(frame_count)):
            torch.cuda.empty_cache()
            success, frame = video.read()

            # Image preprocessing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess(frame)
            frame = frame.unsqueeze(0).to(device)

            # Feed Through Model
            frame = transformer(frame)

            # Image Deprocessing
            frame = frame.squeeze()
            frame = frame.cpu().clamp(0, 255).numpy()
            frame = frame.transpose(1, 2, 0)
            frame = np.uint8(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Outputs image
            if args.show_frame == True:
                cv2.imshow('Style Cam', frame)
            vout.write(frame)

            if cv2.waitKey(1) == 27:
                break

    # Release everything after we're finished
    video.release()
    vout.release()
    cv2.destroyAllWindows()
