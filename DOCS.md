# style.py flags
style.py uses subparsers to run different submodules

Template:
```bash
python style.py {subparser} {{flags}}
```
## Training
### Subparser: `train`
### Flags
* `--dataset`: Path to Training images folder
* `--style`: Path to the style image
* `--image-size`: Size of training images, default is 256
* `--content-weight`: Weight for Content Loss default is 2e2
* `--style-weight`: Weight for Style Loss
* `--tv-weight`: Weight for TV Loss, default is 1e-6
* `--norm-range`: Normalization range for VGG input, True for (0,255) & False for (0,1). Default is True.
* `--epochs`: Number of iterations through the dataset, default is 2
* `--batch-size`: Batch Size for Training. Default is 4.
* `--log-interval`: Interval for Logging a Training Iteration. Default is 2.
* `--save-interval`: How often it saves a copy of the training weights
* `--checkpoint-dir`: Where it saves the training weights as a `.pth`. Default is models
  
## Image Stylization
### Subparser: `eval`
### Flags
* `--content`: Path to content image
* `--out`: Path to to output image. Example: `--out images/output.jpg`
* `--model`: Path to model

## Webcam Stylization
### Subparser: `cam`
### Flags
* `--model`: Path to model
* `--width`: Width of webcam output in pixel, not required
* `--height`: Height of webcam output in pixel, not required

## Video Stylization
### Subparser: `video`
### Flags
* `--model`: Path to model
* `--content`: Path to content video
* `--output-dir`: Path to output video, e.g. `images/results`
* `--show-frame`: Show frames as they're being processed, default is False

## Export to Onnyx
### Subparser: `export`
### Flags
* `--model`: Path to model
* `--output-dir`: Path to output, default is models