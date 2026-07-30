# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


logger = logging.get_logger(__name__)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaLN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.silu(self.linear(x))



def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# TODO: replace with torch native equivalent
def pad_sequences(seqs, pad_value):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    padded = [
        s + [pad_value] * (max_len - len(s))
        for s in seqs
    ]

    return padded, lengths


def normalize_transition_schedule(transition):
    if transition is None:
        return None
    if isinstance(transition, (DictConfig, ListConfig)):
        transition = OmegaConf.to_container(transition, resolve=True)

    if isinstance(transition, dict):
        schedule = [
            (int(block_idx), int(epochs))
            for block_idx, epochs in transition.items()
        ]
    else:
        schedule = [(int(block_idx), int(epochs)) for block_idx, epochs in transition]

    if not schedule:
        raise ValueError("transition must contain at least one block schedule entry")
    for block_idx, epochs in schedule:
        if epochs <= 0:
            raise ValueError("transition epoch counts must be positive")
    return schedule



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0

        #seq_len = 1 + n_ctx // 2

        #num_noisy = n_ctx - seq_len

        #assert num_noisy == seq_len - 1
		
		#mask = torch.zeros(n_ctx, n_ctx, dtype=torch.bool)
		#
		## Clean token rows:
		## x_i can attend to x_1 ... x_i.
		#mask[:seq_len, :seq_len] = torch.tril(
		#    torch.ones(seq_len, seq_len, dtype=torch.bool)
		#)
		#
		#for i in range(num_noisy):
		#    noisy_position = seq_len + i
		#
		#    mask[noisy_position, : i + 1] = True
		#    mask[noisy_position, noisy_position] = True
		#
		#self.register_buffer(
		#    "bias",
		#    mask.view(1, 1, n_ctx, n_ctx),
		#)

        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v, masks, inference_mode=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = masks[:,None,:,:].to(device) #self.bias[:, :, ns-nd:ns, :ns]
        b = b.expand_as(w)
        #print(b)
        w = w * b - 1e20 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        print("Attention weights")
        #print(w)
        if inference_mode:
            print("Attention weights :")
            # print(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    
    # single not batched
    def derive_masks(self, original_mask, noise_mask, inference_mode=False):

        #print(original_mask)
        #print(noise_mask)

        mask_dim_ =  original_mask.shape[-1]
        mask = torch.zeros(mask_dim_, mask_dim_)

        if inference_mode:
            print("Inference mode mask")
            mask = torch.zeros(mask_dim_, mask_dim_)
            mask[1:,1:] = torch.tril(torch.ones(mask_dim_ - 1, mask_dim_ - 1))
            #print(mask)
            return mask

        i = int(original_mask.sum())
        print(f"i IS {i}")
        ni = int(noise_mask.sum()) - 1

        num_noisy = int(noise_mask.sum()) - 1

        # steps:
        #mask[0:i,0] = True
        #mask[:i, :i] = torch.tril(
		#    torch.ones(i, i, dtype=torch.bool)
		#)

        mask[i+1:i+1+ni, :ni-1] = torch.tril(torch.ones(ni-1,ni-1, dtype=torch.bool))

        for z in range(num_noisy):

            noisy_position = i + z
            #mask[noisy_position, : z] = True
            #mask[noisy_position, 1 + z] = False
            mask[noisy_position, noisy_position] = True

        print("training mode mask")
        #print(mask)

        return mask


    def forward(self, x, original_masks, noise_masks, layer_past=None, inference_mode=False):
        print(f"-----{x.shape}")
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        # Just mask construction
        masks = []
        for i in range(x.shape[0]):
            mask = self.derive_masks(original_masks[i], noise_masks[i], inference_mode=inference_mode)
            masks.append(mask)
        masks = torch.stack(masks, axis=0)

        a = self._attn(query, key, value, masks, inference_mode=inference_mode)
        # print(torch.linalg.vector_norm(a,dim=-1))
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.adaLN_modulation = AdaLN(config.cond_hidden_size, 6 * nx, bias=True)
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, inputs, conditioning, layer_past=None, inference_mode=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(6, dim=1)

        
        x = inputs["seqs"]
        x = self.ln_1(x)
        x = modulate(x, shift_msa, scale_msa)
        a, present = self.attn(x, inputs["original_mask"], inputs["loss_mask"], layer_past=layer_past, inference_mode=inference_mode)
        a = gate_msa.unsqueeze(1) * a
        x = x + a
        x = self.ln_2(x)
        x = modulate(x, shift_mlp, scale_mlp)
        m = self.mlp(x)
        m = gate_mlp.unsqueeze(1) * m
        x = x + m
        inputs["seqs"] = x
        return inputs, present


class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # what is cond_hidden_size 
        self.adaLN_modulation = AdaLN(
                config.cond_hidden_size, 2 * config.n_embd, bias=True
            )

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def get_embeddings(self, input_ids):
        #input_shape = input_ids.size()
        # input_ids = input_ids.view(-1, input_ids.size(-1))

        # NOTE: 0 hardcoded instead of past_length
        #position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long,
        #                                device=input_ids.device)
        #position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        #position_ids = position_ids.view(-1, position_ids.size(-1))
        inputs_embeds = self.wte(input_ids)
        #position_embeds = self.wpe(position_ids)

        return inputs_embeds #+ position_embeds

    # NOTe: input
    def add_position_embeddings(self, inputs, offset=0):
        # expected input shape is [B, S, Hidden_dim]
        position_ids = torch.arange(offset, inputs.size(0) + offset, dtype=torch.long, device=inputs.device) # [seq]
        #position_ids = position_ids.unsqueeze(-2).expand_as(inputs) # Note: this is to 
        #position_ids = position_ids.view(-1, position_ids.size(-1))
        position_embeds = self.wpe(position_ids)
        return position_embeds + inputs


    # NOTE: embeddings expected to be done in prior (As opposed to the original implementation)
    def forward(self, embeds, layer_indices, position_ids=None, token_type_ids=None, past=None, conditioning=None, inference_mode=False):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        #if position_ids is None:
        #    position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
        #                                device=input_ids.device)
        #    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)


        # 
        #if token_type_ids is not None:
        #    token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        #    token_type_embeds = self.wte(token_type_ids)
        #else:
        #    token_type_embeds = 0
        #hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = embeds
        presents = []
        for layer_index, (block, layer_past) in enumerate(zip(self.h, past)):
            if layer_index not in layer_indices:
                continue
            hidden_states, present = block(hidden_states, conditioning, layer_past, inference_mode=inference_mode)
            presents.append(present)

        original = hidden_states 
        hidden_states = hidden_states["seqs"] 
        hidden_states = self.ln_f(hidden_states)
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        hidden_states = modulate(hidden_states, shift, scale)
        #output_shape = input_shape + (hidden_states.size(-1),)
        #original["seqs"] = hidden_states.view(*output_shape)
        original["seqs"] = hidden_states
        return original, presents


class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            cond_hidden_size = 128,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.cond_hidden_size = cond_hidden_size


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        self.config = config
        super(GPT2LMHeadModel, self).__init__()
        self.time_embedder = TimestepEmbedder(config.cond_hidden_size)
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.adaLN_modulation = AdaLN(
                config.cond_hidden_size, 2 * config.n_embd, bias=True
            )

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    # labels = [hello world], [hello world <noisy world>]
    def forward(self, embeds, timesteps, layer_indices, position_ids=None, token_type_ids=None, lm_labels=None, past=None, sigma_stuff=None, inference_mode=False):
        conditioning = F.silu(self.time_embedder(timesteps.to(device)))
        outputs, presents = self.transformer(embeds, layer_indices, None, None, None, conditioning=conditioning, inference_mode=inference_mode)
        #hidden_states = torch.nn.utils.rnn.pad_sequence([x[y.bool()] for (x,y) in zip(outputs["seqs"],outputs["loss_mask"])], batch_first=True)
        #hidden_states = outputs["seqs"][outputs["loss_mask"].bool()]
        c_out, zt, c_skip = sigma_stuff


        #if hidden_states.size()[1] == 1:
        #    pass
        #else:
        #    hidden_states = hidden_states[:,1:]

        hidden_states = outputs["seqs"]

        hidden_states = (hidden_states * c_out[:, None, None] + zt * c_skip[:, None, None]).float()

        lm_logits = self.lm_head(hidden_states)

        #lm_logits = lm_logits[outputs["loss_mask"]]
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents
