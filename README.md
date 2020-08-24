# Neural-Artist
Changes include: L2 loss swapped with L1

## Setup Training
Using 2014 COCO Training images
On ubuntu run setup_train.sh

```python style.py train --style images/style/wave_2.jpg --dataset data --checkpoint-dir models```


## Videos
<p align = 'center'>K-pop is in, right?</p>
<p align = 'center'>
<img src = 'images/style/cubist.jpg' height = '246px'>
<img src = 'images/content/blackpink_before.gif' height = '246px'>
<a href = 'images/results/blackpink_cubist.gif'><img src = 'images/results/blackpink_cubist.gif' width = '689px'></a>
</p>
<p align = 'center'>
It took 22 minutes on a GTX 1080 to style the full (1920x1080) video by <a href = 'https://www.youtube.com/watch?v=32si5cfrCNc'>Black Pink</a>. Full video <a href = 'https://drive.google.com/file/d/1HSOVhgkP1omsjxhPrXpJxRWzSaR24Tss/view?usp=sharing'>here</a>.
</p>

### Evaluating Models on Videos

You can produce a simple mp4 video using the following command:
```bash
python style.py video --model {path to model} --content {path to content video} --output-dir {output path} --show-frame {True/False, shows current Frame being processed}
```
Example:
```bash
python style.py video --model models/style_cubist.pth --content images/content/blackpink.mp4 --output-dir images/results
```

### Audio
Audio isn't processed with opencv, so you'll have to use [ffmpeg](https://ffmpeg.org/) if you want audio added to your video.
```
ffmpeg -i {input mp4} -i {input audio mp4 or mp3} -c copy output.mp4
```

Example:
```
ffmpeg -i images/results/blackpink.mp4 -i audio.mp3 -c copy output.mp4
```

## Webcam

# INSERT GIFS

### Evaluating Models using Webcams
You can produce live webcam styling using the following:


## Acknowledgements
* Original Algorithm by Leon A. Gatys: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* Feedforward method developed by Justin Johnson: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/)
* [Demystifying Neural Style Transfer](https://arxiv.org/pdf/1701.01036.pdf)
* Improved Fast Stylization using [Instance Normalization](https://arxiv.org/abs/1607.08022) by Dmitry Ulyanov
* [Pytorch example library](https://github.com/pytorch/examples/blob/master/fast_neural_style) I used for debugging(Normalizing input completely flew over my head 🙃)
* I got the cubist painting from [hzy46](https://github.com/hzy46/fast-neural-style-tensorflow), I don't know where it's from unfortunately.
* README styling influenced by [Lengstrom](https://github.com/lengstrom)