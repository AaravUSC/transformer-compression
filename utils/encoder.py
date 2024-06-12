import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio
import numpy as np

import logging
from typing import List, Optional, Tuple
_LG = logging.getLogger(__name__)

import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter

class LayerNorm(nn.LayerNorm):
  

    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x

class FeatureProjection(Module):
  

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(
            in_features,
            out_features,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
      
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class ConvolutionalPositionalEmbedding(Module):
   
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        for hook in self.conv._forward_pre_hooks.values():
            if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                _LG.warning("Removing weight_norm from %s", self.__class__.__name__)
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x):
        
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class SelfAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.head_dim = head_dim

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        #position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        # scale down q to avoid value overflow.
        weights = (self.scaling * q) @ k  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask
        weights = weights - weights.max(dim=-1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v  # B, nH, L, Hd
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output, None  # Necessary for compatibility with WavLMSelAttention


class FeedForward(Module):
    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
       
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


def _compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> Tensor:


    batch_size, frame = shape
    mask = torch.full((batch_size, frame), False)
    # add a random number for probabilistic rounding
    all_num_mask = int(mask_prob * frame / float(mask_length) + torch.rand(1))

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(batch_size):
        if padding_mask is not None:
            sz = frame - padding_mask[i].long().sum().item()
            # add a random number for probabilistic rounding
            num_mask = int(mask_prob * sz / float(mask_length) + torch.rand(1))
            num_mask = max(min_masks, num_mask)
        else:
            sz = frame
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = torch.full((num_mask,), mask_length)
        elif mask_type == "uniform":
            lengths = torch.randint(mask_other, mask_length * 2 + 1, size=(num_mask,))
        elif mask_type == "normal":
            lengths = torch.normal(mask_length, mask_other, size=(num_mask,))
            lengths = torch.maximum(torch.ones(1), torch.round(lengths)).int()
        elif mask_type == "poisson":
            lengths = torch.poisson(mask_length, size=(num_mask,))
            lengths = torch.round(lengths).int()
        else:
            raise Exception(f"unknown mask selection: {mask_type}")

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = torch.randint(s, e - length, size=(1,))
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = torch.tensor([e - s for s, e in parts], dtype=torch.int)
                lens[lens < length + min_space] = 0
                l_sum = lens.sum()
                if l_sum == 0:
                    break
                probs = lens / l_sum
                c = torch.distributions.categorical.Categorical(probs).sample()
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = torch.tensor(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = torch.randperm(sz - min_len)[:num_mask]
            mask_idc = torch.tensor(
                [mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])]
            )

        mask_idcs.append(torch.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = mask_idc[torch.randperm(len(mask_idc))[:min_len].long()]
        mask[i, mask_idc] = True

    return mask


def _get_padding_mask(input: Tensor, lengths: Tensor) -> Tensor:
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        input (Tensor): The padded Tensor of dimension `[batch, max_len, frequency]`.
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.
    Returns:
        (Tensor): The padding mask.
    """
    batch_size, max_len, _ = input.shape
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
    return mask


class MaskGenerator(Module):

    def __init__(
        self,
        encoder_embed_dim: int,
        mask_prob: float,
        mask_selection: str,
        mask_other: float,
        mask_length: int,
        no_mask_overlap: bool,
        mask_min_space: int,
        mask_channel_prob: float,
        mask_channel_selection: str,
        mask_channel_other: float,
        mask_channel_length: int,
        no_mask_channel_overlap: bool,
        mask_channel_min_space: int,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.mask_length = mask_length
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.mask_channel_length = mask_channel_length
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        self.mask_embedding = Parameter(torch.FloatTensor(encoder_embed_dim))
        torch.nn.init.uniform_(self.mask_embedding)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = _compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = mask_indices.to(x.device)
            x[mask_indices] = self.mask_embedding
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = _compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = mask_channel_indices.to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0

        return x, mask_indices


def _compute_logits(
    proj_x: Tensor,
    target: Tensor,
    label_embeddings: Parameter,
) -> Tensor:
    logit_temp = 0.1
    pos = torch.index_select(label_embeddings, 0, target.long())
    negs = label_embeddings.unsqueeze(1).expand(-1, proj_x.size(0), -1)
    neg_is_pos = (pos == negs).all(-1)
    pos = pos.unsqueeze(0)
    targets = torch.cat([pos, negs], dim=0)

    logits = torch.cosine_similarity(proj_x.float(), targets.float(), dim=-1).type_as(proj_x)
    logits /= logit_temp
    if neg_is_pos.any():
        logits[1:][neg_is_pos] = float("-inf")
    logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
    return logits


class LogitGenerator(Module):

    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
        skip_masked: bool,
        skip_nomask: bool,
    ):
        super().__init__()
        self.label_embeddings = Parameter(torch.FloatTensor(num_classes, final_dim))
        torch.nn.init.uniform_(self.label_embeddings)
        self.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim)
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask

    def forward(self, x: Tensor, label: Tensor, mask_m: Tensor, mask_u: Tensor) -> Tuple[Tensor, Tensor]:
        proj_x = self.final_proj(x)
        if self.skip_masked:
            logit_m = None
        else:
            proj_x_m = proj_x[mask_m]
            label_m = label[mask_m]
            logit_m = _compute_logits(proj_x_m, label_m, self.label_embeddings)

        if self.skip_nomask:
            logit_u = None
        else:
            proj_x_u = proj_x[mask_u]
            label_u = label[mask_u]
            logit_u = _compute_logits(proj_x_u, label_u, self.label_embeddings)
        return logit_m, logit_u


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

#Compressed Feedforward Module
class FeedForwardC(Module):
    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
        t:int
    ):

      super().__init__()
      self.intermediate_dense_1 = nn.Linear(io_features, int(io_features/t), bias=False)
      self.intermediate_dense_2 = nn.Linear(int(io_features/t), intermediate_features, bias=True)
      self.intermediate_dropout = nn.Dropout(intermediate_dropout)
      self.output_dense = nn.Linear(intermediate_features, io_features)
      self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):

        x = self.intermediate_dense_1(x)
        x = self.intermediate_dense_2(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x

#Compressed Self Attention Module
class SelfAttentionC(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        t:int,
        dropout: float = 0.0,

    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.head_dim = head_dim
        self.scaling = self.head_dim**-0.5

        self.k_proj_1 = nn.Linear(embed_dim, int(embed_dim/t), bias=False)
        self.k_proj_2 = nn.Linear(int(embed_dim/t), embed_dim, bias=True)
        self.v_proj_1 = nn.Linear(embed_dim, int(embed_dim/t), bias=False)
        self.v_proj_2 = nn.Linear(int(embed_dim/t), embed_dim, bias=True)
        self.q_proj_1 = nn.Linear(embed_dim, int(embed_dim/t), bias=False)
        self.q_proj_2 = nn.Linear(int(embed_dim/t), embed_dim, bias=True)


        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        #position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):

        batch_size, length, embed_dim = x.size()

        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")



        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj_1(x)
        q = self.q_proj_2(q).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        k = self.k_proj_1(x)
        k = self.k_proj_2(k).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L

        v = self.v_proj_1(x)
        v = self.v_proj_2(v).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        # scale down q to avoid value overflow.
        weights = (self.scaling * q) @ k  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask


        weights = weights - weights.max(dim=-1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(weights, dim=-1)

        weights = self.dropout(weights)

        output = weights @ v  # B, nH, L, Hd
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output, None  # Necessary for compatibility with WavLMSelAttention

class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        attention: Module,
        dropout: float,
        layer_norm_first: bool,
        feed_forward: Module,
    ):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        #position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
     
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)
        x, _ = self.attention(
            x, attention_mask=attention_mask, #position_bias=position_bias,
            key_padding_mask=key_padding_mask
        )

        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x, None


class Transformer(Module):
    def __init__(
        self,
        pos_conv_embed: Module,
        dropout: float,
        layers: Module,
        layer_norm_first: bool,
        layer_drop: float,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: Tensor):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        return x

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        #position_bias: Optional[Tensor] = None
    ) -> Tensor:

        x = self._preprocess(x)

        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):

                x, _ = layer(x, attention_mask, #position_bias=position_bias
                         )

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(f"`num_layers` must be between [1, {len(self.layers)}]")

        ret: List[Tensor] = []
        x = self._preprocess(x)
        for layer in self.layers:
            x = layer(x, attention_mask)  # Ignore position_bias
            print(x)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret


class Encoder(Module):
    def __init__(
        self,
        feature_projection: Module,
        transformer: Module,
    ):
        super().__init__()
        self.feature_projection = feature_projection
        self.transformer = transformer

    def _preprocess(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)

        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            # extend the mask to attention shape and set weight
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask
    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:

        x, mask = self._preprocess(features, lengths)

        x = self.transformer(x, attention_mask=mask)
        return x

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        x, masks = self._preprocess(features, lengths)
        return self.transformer.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)


class Wav2Vec2Model(Module):

    def __init__(
        self,
        feature_extractor: Module,
        encoder: Module,
        aux: Optional[Module] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:

        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
    
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x



def _get_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    num_heads: int,
    attention_dropout: float,
    ff_interm_features: int,
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
    lays_lis
) -> Encoder:

    feature_projection = model.encoder.feature_projection
    pos_conv = model.encoder.transformer.pos_conv_embed

    encoder_layers = nn.ModuleList()
    for _ in range(num_layers):
     
      encoder_layers.append(lays_lis[_])


    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)
