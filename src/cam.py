"""Stylizes WebCam Images"""
import torch
from torchvision import transforms
import cv2
from src.transformer_net import TransformerNet


def webcam(args):
    """Stylizes webcam footage
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Model
    transformer = TransformerNet()
    state_dict = torch.load(args.model)
    transformer.load_state_dict(state_dict)
    transformer.eval().to(device)

    # Image Transforms Preprocessing
    preprocess = transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor(),
         transforms.Lambda(lambda x: x.mul(255))])

    # Setup webcam
    cam = cv2.VideoCapture(0)
    if args.width is not None:
        cam.set(3, args.width)
    if args.height is not None:
        cam.set(4, args.height)

    # Cam Loop
    with torch.no_grad():
        while True:
            torch.cuda.empty_cache()
            success, frame = cam.read()

            # Image preprocessing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess(frame)
            frame = frame.unsqueeze(0).to(device)

            # Feed Through Model
            frame = transformer(frame)

            # Image deprocessing
            frame = frame.squeeze()
            frame = frame.cpu().numpy()
            frame = frame.transpose(1, 2, 0)
            frame /= 255.0
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow('Style Cam', frame)

            # Press ESC to quit
            if cv2.waitKey(1) == 27:
                break

    cam.release()
    cv2.destroyAllWindows()
