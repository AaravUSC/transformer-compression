# transformer-compression

This repository is an adaptation of the work in [Masana et. al 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Masana_Domain-Adaptive_Deep_Network_ICCV_2017_paper.pdf) to a 12-layer transformer encoder architecture. Domain-adaptive compression is an SVD-based compression technique that optimizes with respect to the outputs of a single forward pass, in theory preserving model parameters that are most relevant to the input data. In this manner, compression is adaptive to the domain of the data used to compress, making this an effective technique for adapting large pre-trained models for domain-subset tasks.  

## Setup

After cloning the repository, install the requirements:
> $ pip install -r requirements.txt

## Compression

This repository applies Domain-Adaptive SVD Compression to [wav2vec2.0](https://arxiv.org/abs/2006.11477) using the Librispeech dataset. The example provided in main.py uses only 20 utterances for compression and tests a single example for the sake of usability on CPU. In practise, using at least 200 utterances to compress is recommended for domain-adaptation.

To compress, run the following command:
> $ python main.py --layers='9 10 11 12' --compression=3 --lambda=0.1

**layers**: A space-separated list of numbers between 1 and 12 (inclusive), indicating the layers of the model that will be compressed
**compression**: A value indicating the degree to which the selected layers will be compressed. Recommended: 3.
**lambda**: A regularization parameter. Recommended: 0.1.
