import numpy as np
import flashlight-text
import logging
from typing import List, Optional, Tuple
_LG = logging.getLogger(__name__)
import argparse
import torch
import torchaudio
from torch import nn, Tensor
from torch.nn import Module, Parameter
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from utils.decoder import GreedyCTCDecoder
import utils.compress

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--layers', type=str,
                      help='Layers to compress (integers from 1-12 separated by space)')
  parser.add_argument('--compression', type=float, nargs="*",
                      help='Degree of compression')
  parser.add_argument('--lambda', type=float,
                      help='Regularization Parameter')
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset = torchaudio.datasets.LIBRISPEECH(root='sample_data', url='train-clean-100', download = True)
  Datloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
  bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
  print("Sample Rate:", bundle.sample_rate)
  print("Labels:", bundle.get_labels())
  model = bundle.get_model().to(device)
  actual = [];
  waveforms = [];
  count = 0;
  #only consider a subset of waveforms for now
  while(count < 20):
    waveforms.append(dataset[count][0])
    actual.append(dataset[count][2])
    count +=1
  layers = args.layers.split()
  wav2v = get_compressed_model(waveforms, model, layers, args.compression, args.lambda)
  #Test one example:
  ex = dataset[50][0]
  emiss = wav2v(ex)
  decoder = GreedyCTCDecoder(labels = bundle.get_labels())
  print(decoder(emiss))
