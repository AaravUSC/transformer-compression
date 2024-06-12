import torch
import torchaudio
import numpy as np
import utils.encoder
import utils.transformations

def extract_activations(waveforms, model):
  max = 0
  for x in waveforms:
    if x.size()[1] > max:
      max = x.size()[1]
  for x in waveforms:
    padded_waveforms.append(torch.nn.functional.pad(x, (0, max-x.size()[1])))
    emissions = []
  #feed forward one-at-a-time for compatibility with low-RAM runtimes
  for waveform in padded_waveforms:
    with torch.inference_mode():
      emission, _ = model.feature_extractor(waveform, length = 768)
      emissions.append(emission)
  activations = emissions[0][0]
  for i in range(1,len(emissions)):
    act = emissions[i][0]
    activations = torch.cat((activations,act))
  activations = activations.reshape((len(emissions), int(activations.size()[0]/len(emissions)), 512))
  return activations

def get_encoder_layers(model):
  Encoder_layer_lis = []
  for c in model.children():
    for x in c.children():
      for m in x.children():
        for a in m.children():
          Encoder_layer_lis.append(a)
  Encoder_layer_lis = Encoder_layer_lis[9:]
  return Encoder_layer_lis

def Compression_Feedforward(acts,layer, linlist, biaslist):
  #linear1 = compress_DALR(layer.feed_forward.intermediate_dense, acts, layer.feed_forward.intermediate_dense.weight, layer.feed_forward.intermediate_dense.bias,int(layer.feed_forward.intermediate_dense.weight.size()[1]/degree_of_compression),lamb)
  linear1 = compress_SVD(layer.feed_forward.intermediate_dense, int(layer.feed_forward.intermediate_dense.weight.size()[1]/degree_of_compression))
  int_dense_1 = torch.nn.Linear(in_features = 768, out_features = int(768/degree_of_compression), bias = False)
  int_dense_2 = torch.nn.Linear(in_features = int(768/degree_of_compression), out_features = 3072, bias = True)
  with torch.no_grad():
    int_dense_1.weight.copy_(torch.from_numpy(linear1[0]))
    int_dense_2.weight.copy_(torch.from_numpy(linear1[1]))
    int_dense_2.bias.copy_(layer.feed_forward.intermediate_dense.bias)
  activs = layer.feed_forward.intermediate_dense(acts)
  activs = torch.nn.functional.gelu(activs)
  activs = layer.feed_forward.output_dense(activs)
  biaslist.append(layer.feed_forward.intermediate_dense.bias)
  linlist.append(linear1)
  return activs

def compress_layers(activations, layers_to_compress, degree_of_compression, model, lamb):
  #lists to store compressed layers
  bias_SA = []
  lin_SA = []
  bias_ff = []
  lin_ff = []
  count = 0
  #The rest of this primarily runs the activations through the model layer-by-layer, and compresses layers as the activations are passed through
  with torch.inference_mode():
    activations, mask = model.encoder._preprocess(activations)
    activations = model.encoder.transformer._preprocess(activations)
    enc_count = 0;
    for lay in model.encoder.transformer.layers:
      enc_count+=1
      residual = activations
      if lay.layer_norm_first:
          activations = lay.layer_norm(activations)
      #Self Attention
      batch_size, length, embed_dim = activations.size()
      shape = (batch_size, length, lay.attention.num_heads, lay.attention.head_dim)
      temp_acts = activations
      if(enc_count in layers_to_compress):
        k_pro = compress_DALR(lay.attention.k_proj, temp_acts, lay.attention.k_proj.weight, lay.attention.k_proj.bias, int(lay.attention.k_proj.weight.size()[1]/degree_of_compression), lamb)
        v_pro = compress_DALR(lay.attention.v_proj, temp_acts, lay.attention.v_proj.weight, lay.attention.v_proj.bias, int(lay.attention.v_proj.weight.size()[1]/degree_of_compression), lamb)
        q_pro = compress_DALR(lay.attention.q_proj, temp_acts, lay.attention.q_proj.weight, lay.attention.q_proj.bias, int(lay.attention.q_proj.weight.size()[1]/degree_of_compression), lamb)
        lin_SA.append(k_pro)
        lin_SA.append(v_pro)
        lin_SA.append(q_pro)
        bias_SA.append(lay.attention.k_proj.bias)
        bias_SA.append(lay.attention.v_proj.bias)
        bias_SA.append(lay.attention.q_proj.bias)
      q = lay.attention.q_proj(temp_acts).view(*shape).transpose(2, 1)
      k = lay.attention.k_proj(temp_acts).view(*shape).permute(0, 2, 3, 1)
      v = lay.attention.v_proj(temp_acts).view(*shape).transpose(2, 1)
      weights = (lay.attention.scaling * q) @ k  # B, nH, L, L
      weights = weights - weights.max(dim=-1, keepdim=True)[0]
      weights = torch.nn.functional.softmax(weights, dim=-1)
      output = weights @ v  # B, nH, L, Hd
      output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)
      activations = lay.attention.out_proj(output)
      activations = lay.dropout(activations)
      activations = residual + activations
      if enc_count in layers_to_compress:
        if lay.layer_norm_first:
          activations = activations + Compression_Feedforward(activations, lay, lin_ff, bias_ff)
        else:
          activations = lay.layer_norm(activations)
          activations = lay.final_layer_norm(activations + Compression_Feedforward(activations, lay, lin_ff, bias_ff))
      else:
        if lay.layer_norm_first:
            activations = activations + lay.feed_forward(lay.final_layer_norm(activations))
        else:
            activations = lay.layer_norm(activations)
            activations = lay.final_layer_norm(activations + lay.feed_forward(activations))
    if not model.encoder.transformer.layer_norm_first:
      activations = model.encoder.transformer.layer_norm(activations)
  i = 0
  #Custom encoder layer list - using compressed layers for the encoder blocks specified, and otherwise using encoder blocks directly from the original wav2vec 2.0
  Encoder_layer_lis_C = []
  for j in range(1,13):
    if j not in layers_to_compress:
      Encoder_layer_lis_C.append(Encoder_layer_lis[j-1])
    else:
      with torch.no_grad():
        Attn_blc = SelfAttentionC(embed_dim=768,num_heads=12,t=degree_of_compression)
        Attn_blc.k_proj_1.weight.copy_(torch.from_numpy(lin_SA[3*i][0]))
        Attn_blc.k_proj_2.weight.copy_(torch.from_numpy(lin_SA[3*i][1]))
        Attn_blc.k_proj_2.bias.copy_(bias_SA[3*i])
        Attn_blc.q_proj_1.weight.copy_(torch.from_numpy(lin_SA[3*i+1][0]))
        Attn_blc.q_proj_2.weight.copy_(torch.from_numpy(lin_SA[3*i+1][1]))
        Attn_blc.q_proj_2.bias.copy_(bias_SA[3*i+1])
        Attn_blc.v_proj_1.weight.copy_(torch.from_numpy(lin_SA[3*i+2][0]))
        Attn_blc.v_proj_2.weight.copy_(torch.from_numpy(lin_SA[3*i+2][1]))
        Attn_blc.v_proj_2.bias.copy_(bias_SA[3*i+2])
        Feed_forward_block = FeedForwardC(io_features=768, intermediate_features=3072, intermediate_dropout=0.0, output_dropout=0.0, t = degree_of_compression)
        Feed_forward_block.intermediate_dense_1.weight.copy_(torch.from_numpy(lin_ff[i][0]))
        Feed_forward_block.intermediate_dense_2.weight.copy_(torch.from_numpy(lin_ff[i][1]))
        Feed_forward_block.intermediate_dense_2.bias.copy_(bias_ff[i])
      Encoder_blc = EncoderLayer(attention=Attn_blc,dropout=0.0,layer_norm_first=False,feed_forward=Feed_forward_block)
      Encoder_layer_lis_C.append(Encoder_blc)
      i+=1
  return Encoder_layer_lis_C


def get_compressed_model(waveforms, model, layers_to_compress, degree_of_compression, lamb):
  activations = extract_activations(waveforms, model)
  encoder_layers = get_encoder_layers(model)
  Encoder_layer_lis_C = compress_layers(activations, layers_to_compress, degree_of_compression, model, lamb)
  Enc = _get_encoder(in_features=768,
      embed_dim=512,
      dropout_input=0.0,
      pos_conv_kernel=128,
      pos_conv_groups=16,
      num_layers=12,
      num_heads=12,
      attention_dropout=0.0,
      ff_interm_features=3072,
      ff_interm_dropout=0.0,
      dropout=0.0,
      layer_norm_first=True,
      layer_drop=0.0,
      lays_lis=Encoder_layer_lis_C)
  #Copy over remaining weights
  with torch.no_grad():
    for x in model.encoder.state_dict():
      if x in Enc.state_dict():
        Enc.state_dict()[x].copy_(model.encoder.state_dict()[x])
    Enc.transformer.layer_norm.weight.copy_(model.encoder.transformer.layer_norm.weight)
    Enc.transformer.layer_norm.bias.copy_(model.encoder.transformer.layer_norm.bias)
  wav2v = Wav2Vec2Model(feature_extractor=model.feature_extractor, encoder = Enc, aux = model.aux)
  return wav2v
