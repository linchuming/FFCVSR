# FFCVSR (AAAI 2019)
AAAI 2019 paper "Frame and Feature-Context Video Super-Resolution" [1]  
[Paper](FFCVSR.pdf)  

### Code

We release the new FFCVSR with optical flow enhanced version (better performance than original FFCVSR)

The environment is:

- TensorFlow >= 1.6
- Anaconda 3

The model ckpt can be download by google driver: https://drive.google.com/open?id=1yLZVNSPxbqZF0UjeHij2HyH-U2u1M3N4

Run script and get the results: `python test_vid4.py`

### Abstract
For video super-resolution, current state-of-the-art approaches either process multiple low-resolution (LR) frames to produce each output high-resolution (HR) frame separately in a sliding window fashion or recurrently exploit the previously estimated HR frames to super-resolve the following frame. The main weaknesses of these approaches are: 1) separately generating each output frame may obtain high-quality HR estimates while resulting in unsatisfactory flickering artifacts, and 2) combining previously generated HR frames can produce temporally consistent results in the case of short information flow, but it will cause significant jitter and jagged artifacts because the previous super-resolving errors are constantly accumulated to the subsequent frames.   
In this paper, we propose a fully end-to-end trainable frame and feature-context video super-resolution (FFCVSR) network that consists of two key sub-networks: local network and context network, where the first one explicitly utilizes a sequence of consecutive LR frames to generate local feature and local SR frame, and the other combines the outputs of local network and the previously estimated HR frames and features to super-resolve the subsequent frame. Our approach takes full advantage of the inter-frame information from multiple LR frames and the context information from previously predicted HR frames, producing temporally consistent high-quality results while maintaining real-time speed by directly reusing previous features and frames. Extensive evaluations and comparisons demonstrate that our approach produces state-of-the-art results on a standard benchmark dataset, with advantages in terms of accuracy, efficiency, and visual quality over the existing approaches.
### Citation
```
[1]  @inproceedings{ffcvsr,
         author = {Bo Yan, Chuming Lin, and Weimin Tan},
         title = {Frame and Feature-Context Video Super-Resolution},
         booktitle = {AAAI},
         year = {2019}
     }
```
