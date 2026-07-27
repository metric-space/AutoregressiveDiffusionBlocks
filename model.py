import copy
import math
import random
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
from transformers import get_scheduler
from omegaconf import DictConfig, ListConfig, OmegaConf

from vit import load_vit2, AdaLN, TimestepEmbedder, modulate
from dblock_modules import get_block_sigmas, get_discrete_sigmas
from model import ViTModel
from data import get_encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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




class ViTDBlockExperimentModel(ViTModel):
    def __init__(self, args):
        super().__init__(args)
        self.gamma = args.gamma
        self.sigma_data = 0.5
        self.cfg_scale = args.cfg_scale
        self.class_dropout_prob = (
            args.class_dropout_prob if self.cfg_scale > 0.0 else 0.0
        )
        self.num_inference_steps = self.args.num_inference_steps or self.args.num_blocks
        self.block_sigmas = get_block_sigmas(num_layers=self.args.num_blocks)
        self.layer_assignment = None
        self.num_layers = args.num_layers
        self.register_buffer(
            "sigmas",
            get_discrete_sigmas(num_steps=self.num_inference_steps, dblock=True).to(
                self.device
            ),
        )
        self.save_hyperparameters(args)

        self.transition = normalize_transition_schedule(args.get("transition", None))
        if self.transition is not None:
            for block_idx, _ in self.transition:
                if not 0 <= block_idx < self.args.num_blocks:
                    raise ValueError(
                        f"transition block index {block_idx} is outside "
                        f"range [0, {self.args.num_blocks})"
                    )
        if self.transition is None:
            self.transition_idx = None
            self.current_training_block = random.choice(range(self.args.num_blocks))
            self.training_counter = 1
            self.minimum_block_idx = 0
        else:
            self.transition_idx = 0
            self.args.num_epochs = sum(epochs for _, epochs in self.transition)
            self.current_training_block, self.training_counter = self.transition[0]
            self.minimum_block_idx = min([k for k,_ in self.transition])

    def on_fit_start(self):
        self.logger.watch(
                self,
                log="gradients",
                log_freq=200,
                log_graph=False,
        )


    def configure_model(self):
        self.model = load_vit2(
            image_size=self.image_size, num_labels=self.num_labels, num_layers=self.num_layers, is_dblock=True
        )
        print(self.model)

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def get_embeds(
        self, input_ids: torch.Tensor, is_input: bool = True
    ) -> torch.Tensor:
        if is_input:
            embeds = self.model.get_input_embeddings()(input_ids)
        else:
            embeds = F.embedding(
                input_ids, weight=self.model.get_output_embeddings().weight
            )
        return self.normalize_embeddings(embeds)


    def get_weights(self, sigmas):
        return (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data) ** 2

    def get_sigmas(
        self,
        block_idx: int,
        n_samples: int,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
        sigma_min_block = self.block_sigmas[block_idx]
        sigma_max_block = self.block_sigmas[block_idx + 1]
        if self.gamma > 0.0:
            log_sigma_min = np.log(sigma_min_block)
            log_sigma_max = np.log(sigma_max_block)
            log_range = log_sigma_max - log_sigma_min
            sigma_min_block = np.exp(log_sigma_min - self.gamma * log_range)
            sigma_max_block = np.exp(log_sigma_max + self.gamma * log_range)
            sigma_min_block = max(sigma_min_block, self.block_sigmas[0])
            sigma_max_block = min(sigma_max_block, self.block_sigmas[-1])

        cdf_min_block = norm.cdf((np.log(sigma_min_block) - p_mean) / p_std)
        cdf_max_block = norm.cdf((np.log(sigma_max_block) - p_mean) / p_std)
        rand = np.random.uniform(cdf_min_block, cdf_max_block, n_samples)
        sigma = np.exp(p_mean + p_std * norm.ppf(rand))
        return torch.from_numpy(sigma)

    def on_train_epoch_end(self):
        self.training_counter -= 1

        if self.training_counter <= 0:
            if self.transition is None:
                self.current_training_block = random.choice(range(self.args.num_blocks))
                self.training_counter = 1
            else:
                self.transition_idx += 1
                if self.transition_idx < len(self.transition):
                    self.current_training_block, self.training_counter = self.transition[
                        self.transition_idx
                    ]

    def estimate_target_layer(self, sigma: torch.Tensor) -> int:
        block_sigmas = torch.tensor(self.block_sigmas, device=sigma.device)
        block_idx = torch.bucketize(sigma, block_sigmas, right=True) - 1
        block_idx = (self.args.num_blocks - 1) - block_idx
        block_idx = torch.clamp(block_idx, 0, self.args.num_blocks - 1).long()
        values, counts = block_idx.unique(return_counts=True)
        return max(values[counts.argmax()].item(), self.minimum_block_idx)

    def denoise(self, x, zt, sigma, block_idx=None):
        if block_idx is None:
            block_idx = self.estimate_target_layer(sigma)
        if self.class_dropout_prob > 0.0 and self.training:
            drop_x = torch.rand(x.shape[0], device=x.device) < self.class_dropout_prob
            uncond_x = torch.zeros_like(x)
            x = torch.where(drop_x[:, None, None, None], uncond_x, x)
        elif not self.training and self.cfg_scale > 0.0:
            uncond_x = torch.zeros_like(x)
            x = torch.cat([uncond_x, x])
            zt = torch.cat([zt] * 2)
            sigma = torch.cat([sigma] * 2)

        #self.log("block_idx", block_idx, on_step=True, prog_bar=False)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()

        if self.layer_assignment is None:
            split_size = self.model.config.num_hidden_layers // self.args.num_blocks
            self.layer_assignment = [
                list(range(i * split_size, (i + 1) * split_size))
                for i in range(self.args.num_blocks)
            ]
        outputs = self.model.forward_block(
            layer_indices=self.layer_assignment[block_idx],
            pixel_values=x,
            noisy_embeds=zt * c_in[:, None],
            timesteps=c_noise,
        )
        hidden_states = outputs.last_hidden_state
        conditioning = outputs.conditioning
        model_out = hidden_states * c_out[:, None] + zt * c_skip[:, None]
        logits = self.model.forward_output_embeddings(
            model_out.unsqueeze(1), conditioning
        )
        if not self.training and self.cfg_scale > 0.0:
            logits_uncond, logits_cond = logits.chunk(2)
            logits = logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
        return logits

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        if return_metrics:
            logits = self.diffusion_step(pixel_values)
            if step == "val":
                return self.valid_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif step == "test":
                return self.test_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            else:
                raise NotImplementedError(f"Step {step} is not supported")

        z = self.get_embeds(labels, is_input=True)
        sigmas = self.get_sigmas(self.current_training_block, z.shape[0])
        block_idx = self.estimate_target_layer(sigmas)
        sigmas = sigmas.to(z)
        zt = z + sigmas[:, None] * torch.randn_like(z)
        logits = self.denoise(pixel_values, zt, sigmas, block_idx)
        predicted_token_ids = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(
            logits.view(-1, self.num_labels), labels.view(-1), reduction="none"
        )
        ce_loss = loss.mean()
        w = self.get_weights(sigmas)[:, None]
        loss = (loss * w).mean()

        loss_dict = {
            f"{step}/loss": loss,
            f"{step}/loss_{block_idx}": loss,
            f"{step}/ce_loss": ce_loss,
            f"{step}/ce_loss_{block_idx}": ce_loss,
        }
        return loss, loss_dict

    def diffusion_step(self, x):
        bsz = x.shape[0]
        hidden_size = self.model.config.hidden_size
        z = torch.randn(bsz, hidden_size, device=self.device)
        z *= torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x.new_ones([x.shape[0]])
        for i in range(self.sigmas.shape[0] - 1):
            sigma = self.sigmas[i] * s_in
            next_sigma = self.sigmas[i + 1] * s_in
            # denoise
            logits = self.denoise(x, z, sigma)
            probs = F.softmax(logits, dim=1)
            denoised = F.linear(probs, self.model.get_input_embeddings().weight.t())
            # to d
            d = (z - denoised) / sigma[:, None]
            dt = next_sigma - sigma
            # euler step
            euler_step = z + dt[:, None] * d
            z = euler_step
        min_sigma = self.sigmas[-1].item()
        sigmas = torch.full((x.shape[0],), min_sigma, device=self.device)
        logits = self.denoise(x, z, sigmas)
        return logits

# ------------------------- experimental code ------------------------------------------

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


text_decoder = get_encoder()

class TransformerBlockModel(L.LightningModule): #ViTModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gamma = args.gamma
        self.sigma_data = 0.5
        self.cfg_scale = args.cfg_scale
        self.class_dropout_prob = (
            args.class_dropout_prob if self.cfg_scale > 0.0 else 0.0
        )
        self.num_inference_steps = self.args.num_inference_steps or self.args.num_blocks
        self.block_sigmas = get_block_sigmas(num_layers=self.args.num_blocks)
        self.layer_assignment = None
        self.num_layers = args.num_layers
        self.register_buffer(
            "sigmas",
            get_discrete_sigmas(num_steps=self.num_inference_steps, dblock=True).to(
                self.device
            ),
        )
        self.save_hyperparameters(args)
        self.alternative = args.alternative

        self.transition = normalize_transition_schedule(args.get("transition", None))
        print(self.transition)
        if self.transition is not None:
            for block_idx, _ in self.transition:
                if not 0 <= block_idx < self.args.num_blocks:
                    raise ValueError(
                        f"transition block index {block_idx} is outside "
                        f"range [0, {self.args.num_blocks})"
                    )
        if self.transition is None:
            self.transition_idx = None
            self.current_training_block = random.choice(range(self.args.num_blocks))
            self.training_counter = 1
            self.minimum_block_idx = 0
        else:
            self.transition_idx = 0
            self.args.num_epochs = sum(epochs for _, epochs in self.transition)
            self.current_training_block, self.training_counter = self.transition[0]
            self.minimum_block_idx = min([k for k,_ in self.transition])

    def on_fit_start(self):
        self.logger.watch(
                self,
                log="gradients",
                log_freq=10,
                log_graph=False,
        )


    def configure_model(self):
        self.model = GPT2LMHeadModel(GPT2Config())
        print(self.model)

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    # NOTE: position embeddings should be shifted here?
    def get_embeds(
        self, input_ids: torch.Tensor, is_input: bool = True
    ) -> torch.Tensor:
        if is_input:
            embeds = self.model.get_input_embeddings()(input_ids)
        else:
            embeds = F.embedding(
                input_ids, weight=self.model.get_output_embeddings().weight
            )
        return self.normalize_embeddings(embeds)


    def get_weights(self, sigmas):
        return (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data) ** 2

    def get_sigmas(
        self,
        block_idx: int,
        n_samples: int,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
        sigma_min_block = self.block_sigmas[block_idx]
        sigma_max_block = self.block_sigmas[block_idx + 1]
        if self.gamma > 0.0:
            log_sigma_min = np.log(sigma_min_block)
            log_sigma_max = np.log(sigma_max_block)
            log_range = log_sigma_max - log_sigma_min
            sigma_min_block = np.exp(log_sigma_min - self.gamma * log_range)
            sigma_max_block = np.exp(log_sigma_max + self.gamma * log_range)
            sigma_min_block = max(sigma_min_block, self.block_sigmas[0])
            sigma_max_block = min(sigma_max_block, self.block_sigmas[-1])

        cdf_min_block = norm.cdf((np.log(sigma_min_block) - p_mean) / p_std)
        cdf_max_block = norm.cdf((np.log(sigma_max_block) - p_mean) / p_std)
        rand = np.random.uniform(cdf_min_block, cdf_max_block, n_samples)
        sigma = np.exp(p_mean + p_std * norm.ppf(rand))
        return torch.from_numpy(sigma)

    def on_train_epoch_end(self):
        self.training_counter -= 1

        if self.training_counter <= 0:
            if self.transition is None:
                self.current_training_block = random.choice(range(self.args.num_blocks))
                self.training_counter = 1
            else:
                self.transition_idx += 1
                if self.transition_idx < len(self.transition):
                    self.current_training_block, self.training_counter = self.transition[
                        self.transition_idx
                    ]

    def estimate_target_layer(self, sigma: torch.Tensor) -> int:
        block_sigmas = torch.tensor(self.block_sigmas, device=sigma.device)
        block_idx = torch.bucketize(sigma, block_sigmas, right=True) - 1
        block_idx = (self.args.num_blocks - 1) - block_idx
        block_idx = torch.clamp(block_idx, 0, self.args.num_blocks - 1).long()
        values, counts = block_idx.unique(return_counts=True)
        block_idx = values[counts.argmax()].item()
        if block_idx in self.alternative.keys():
            return self.alternative[block_idx]
        return block_idx

    
    def denoise(self, x, zt, sigma, block_idx=None, inference_mode=False):
        if block_idx is None:
            block_idx = self.estimate_target_layer(sigma)
        if self.class_dropout_prob > 0.0 and self.training:
            drop_x = torch.rand(x.shape[0], device=x.device) < self.class_dropout_prob
            uncond_x = torch.zeros_like(x)
            x = torch.where(drop_x[:, None, None, None], uncond_x, x)
        elif not self.training and self.cfg_scale > 0.0:
            uncond_x = torch.zeros_like(x)
            x = torch.cat([uncond_x, x])
            zt = torch.cat([zt] * 2)
            sigma = torch.cat([sigma] * 2)

        #self.log("block_idx", block_idx, on_step=True, prog_bar=False)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()

        if self.layer_assignment is None:
            split_size = self.model.config.n_layer // self.args.num_blocks
            self.layer_assignment = [
                list(range(i * split_size, (i + 1) * split_size))
                for i in range(self.args.num_blocks)
            ]
        print(f"Layer indices: {self.layer_assignment[block_idx]}")
        outputs = self.model.forward(
                x,
                layer_indices=self.layer_assignment[block_idx],
                timesteps=c_noise,
                sigma_stuff = (c_out.to(device), zt.to(device), c_skip.to(device)),
                inference_mode=inference_mode
        )
        logits, _ = outputs

        # --------- problematic part -------------

        if not self.training and self.cfg_scale > 0.0:
            logits_uncond, logits_cond = logits.chunk(2)
            logits = logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
        return logits

    def get_model_kwargs(self, batch):
        return {}

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):

        # NOTE: make masks here
        processed_batch = []
        labels = []

        #def collate_fn(batch):
        #    end_ = {"mask": [], "seq": []}
        #    for item in batch:
        #        for key_ in ["mask", "seq"]:
        #            end_[key_].append(item[key_])
        #    return end_
        #        

        max_len = max([len(i) for i in batch])

        sigmas = self.get_sigmas(self.current_training_block, len(batch))

        zt = []
        

        # TODO: zip maybe better
        #
        # [  !    start   x2   x3]
        # [ start n2      n3   n4] 
        for index, original_seq in enumerate(batch):
            # embedding

             print(f"Input: {text_decoder.decode(original_seq)}")

             noised_values = original_seq[1:]
             labels.append(torch.tensor([50256] + original_seq[:-1]))

             noised = self.model.transformer.get_embeddings(torch.tensor(noised_values).to(device)) # output shape is [Seq, embed_dim]


             original_len = len(original_seq)

             # NOTE We want to make sure zt_ if of seq length = original (zeroed) + noised

             zt_ = sigmas[index, None].to(device) * torch.randn_like(noised).to(device)
             noised = zt_ + self.model.transformer.add_position_embeddings(noised, offset=1)

             zt.append(torch.cat([torch.zeros(original_len, noised.shape[-1]), zt_], dim=0))

             original = self.model.transformer.get_embeddings(torch.tensor(original_seq).to(device))
             original = self.model.transformer.add_position_embeddings(original)

             # make masks here
             # we need to pad to same length here
             complete_mask = torch.zeros(2*max_len - 1)
             complete_mask[:(2*original_len-1)] = 1

             # doubles as noise_mask
             loss_mask = torch.clone(complete_mask).detach()
             loss_mask[1:original_len] = 0 

             original_mask = torch.zeros(2*max_len - 1)
             original_mask[:original_len] = 1

             # QUESTION: why am I padding here instead of outside the loop?

             padded_seq_ = torch.zeros((2*max_len - 1), 768) # TODO: don't hard code this
             padded_seq_[:(2*original_len -1)] = torch.cat([original, noised], axis=0)

             # final_mask for loss calculation
             # complete_mask
             processed_batch.append({"original_mask": original_mask, "seqs": padded_seq_, "loss_mask": loss_mask, "complete_mask": complete_mask})

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        zt =   torch.nn.utils.rnn.pad_sequence(zt, batch_first=True).to(device)


        block_idx = self.estimate_target_layer(sigmas)
        

        #if return_metrics:
        #    logits = self.diffusion_step(pixel_values)
        #    if step == "val":
        #        return self.valid_metrics(
        #            logits.view(-1, self.num_labels), labels.view(-1)
        #        )
        #    elif step == "test":
        #        return self.test_metrics(
        #            logits.view(-1, self.num_labels), labels.view(-1)
        #        )
        #    else:
        #        raise NotImplementedError(f"Step {step} is not supported")

        # z = self.get_embeds(labels, is_input=True)

        print(zt.shape)


        # ---------- experiemental ----------------------------------------
        collated_processed_batch = {}
        for key_ in ["original_mask", "seqs", "loss_mask", "complete_mask"]:
            v_ = torch.stack([x[key_] for x in processed_batch], axis=0)
            print(f"{key_ } {v_.shape}")
            #assert v_[0].ndim == 3
            collated_processed_batch[key_] = v_.to(device)

        logits = self.denoise(collated_processed_batch, zt, sigmas, block_idx, inference_mode=False)


        # ================ Debugging process =============================

        # NOTE: the mask flattens

        print("Warning! batch dimension is toast here, come back here to correct after testing phase is over")

        logits = logits[collated_processed_batch["loss_mask"].bool()] # should be [seq, hidden_dim]

        predicted_token_ids = torch.argmax(logits, dim=-1)

        print("DEBUG: what is being mapped: ")
        for i in range(len(predicted_token_ids)):
            print(i, text_decoder.decode([predicted_token_ids[i].item()]), "->", text_decoder.decode([labels[0,i].item()]))

        
        #for i in range(predicted_token_ids.shape[0]):
        #    a_ = predicted_token_ids[i]
        #    print(f"Predicted tokens : {text_decoder.decode(a_.tolist())}")
        #    t_ = temp_mask[0]
        #    print(f"Predicted tokens part2 : {text_decoder.decode(a_[t_].tolist())}")
        #    print(f"Expected tokens : {text_decoder.decode(labels[i].tolist())}")


        loss = F.cross_entropy(
                logits.view(-1, 50257), labels[0],
                reduction='none'
        )

        print("loss shape is: ", loss.shape)
        #loss = loss.reshape(-1, loss.size()[-1])
        ce_loss = loss.mean()
        w = self.get_weights(sigmas)[:, None].to(device)
        loss = loss.mean() # (loss * w).mean()

        print(f"LOSS STATEMENT {block_idx} {loss}")

        loss_dict = {
            f"{step}/loss": loss,
            f"{step}/loss_{block_idx}": loss,
            f"{step}/ce_loss": ce_loss,
            f"{step}/ce_loss_{block_idx}": ce_loss,
        }
        return loss, loss_dict


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_scheduler(
            name=self.args.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            scheduler_specific_kwargs=self.args.scheduler_specific_kwargs,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


    def forward(self, **kwargs):
        return self.model(**kwargs).logits


    def training_step(self, batch, batch_idx):
        batch_size = len(batch)
        model_kwargs = self.get_model_kwargs(batch)
        loss, loss_dict = self.shared_step(batch, step="train", **model_kwargs)
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return loss


    #TODO:  this has to be redone
    def diffusion_step(self, original, inference_mode=True):

        # x is input embeddings
        x = original["seqs"]
        #x = self.model.transformer.add_position_embeddings(x)
        bsz = x.shape[0]
        slen = original["loss_mask"].sum().item()
        hidden_size = self.model.config.n_embd
        z = torch.randn(bsz, (slen-1), hidden_size, device=self.device)
        #z = self.model.transformer.add_position_embeddings(z, offset = 1)
        z *= torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x.new_ones([bsz])

        #z = z[:,None,:]

        for i in range(self.sigmas.shape[0] - 1):
            print("Diffusion step called", x.shape, z.shape)
            original["seqs"] = torch.cat([x, z], dim=1).to(device) 
            original["seqs"] = self.model.transformer.add_position_embeddings(original["seqs"])
            sigma = self.sigmas[i] * s_in
            next_sigma = self.sigmas[i + 1] * s_in
            # denoise
            print("> ",original["seqs"].shape)
            logits = self.denoise(original, torch.cat([torch.zeros_like(x),z],dim=1), sigma, inference_mode=inference_mode)[:, -(slen - 1):,:]

            probs = F.softmax(logits, dim=-1)
            denoised = probs @ self.model.transformer.wte.weight

            print("z:", z.shape)
            print("logits:", logits.shape)
            print("probs:", probs.shape)
            print("denoised:", denoised.shape)
            print("argmax:", torch.argmax(logits, dim=-1))
            # to d
            d = (z - denoised) / sigma[:, None] # none is for the hidden_dim
            dt = next_sigma - sigma
            # euler step
            euler_step = z + dt[:, None] * d # None is for the hidden dim
            z = euler_step

        min_sigma = self.sigmas[-1].item()
        original["seqs"] = torch.cat([x, z], dim=1) 
        original["seqs"] = self.model.transformer.add_position_embeddings(original["seqs"])
        sigmas = torch.full((x.shape[0],), min_sigma, device=self.device)
        logits = self.denoise(original,  torch.cat([torch.zeros_like(x),z],dim=1), sigmas, inference_mode=inference_mode)[:, -(slen - 1):, :]
        probs = F.softmax(logits, dim=-1)
        return probs @ self.model.transformer.wte.weight# return logits


    def generate(self, sentence, num_new_tokens, temperature=1.0):
        tokenizer = get_encoder()
        tokenized_words = tokenizer.encode(sentence)
        #tokenized_words = [50256]
        print(tokenized_words)

        list_of_words = sentence.split(" ")

        old_logits = None

        for _ in range(num_new_tokens):

            original = torch.tensor(tokenized_words).to(device)
            original = self.model.transformer.get_embeddings(original).to(device)
            w_ = len(tokenized_words)
            input_ = {"seqs": original.to(device)[None,...], "original_mask": torch.tensor([w_*[1]+[0]]).to(device), "loss_mask": torch.tensor([[1]+(w_-1)*[0]+[1]]).to(device)}

            logits = self.diffusion_step(input_, inference_mode=True)

            if old_logits != None:
                diff = (logits - old_logits).abs()
                print(f"Debug -> {diff.mean()}")
                print("mean:", diff.mean().item())
                print("max: ", diff.max().item())
                print("norm:", diff.norm(dim=-1).mean().item())
            old_logits = logits

            sampling_logits = logits / temperature
            next_token_id = torch.distributions.Categorical(
                    logits=sampling_logits
                    ).sample().detach().item()

            tokenized_words.append(next_token_id)

            next_word = tokenizer.decode([next_token_id])
            print("--------------------------------", next_word)
            print(tokenized_words)
            #list_of_words+=

            #next_embedding = next_embedding[:, None, :]  # [B, 1, D]
            #original = torch.cat([original, next_embedding], dim=1)


        return tokenized_words

    def training_akin_generate(self, sentence, num_new_tokens, temperature=1.0):
        tokenizer = get_encoder()
        tokenized_words = tokenizer.encode(sentence)
        #tokenized_words = [50256]
        print(tokenized_words)

        list_of_words = sentence.split(" ")

        old_logits = None

        for _ in range(num_new_tokens):

            original = torch.tensor(tokenized_words).to(device)
            original = self.model.transformer.get_embeddings(original).to(device)
            w_ = len(tokenized_words)
            print(f"Length of tokenized inference input is : {w_}")
            input_ = {"seqs": original.to(device)[None,...], "original_mask": torch.tensor([w_*[1]+(w_ - 1 + num_new_tokens)*[0]]).to(device), "loss_mask": torch.tensor([[1] + (w_ - 1)*[0]+ (w_ - 1 + num_new_tokens)*[1]]).to(device)}

            logits = self.diffusion_step(input_, inference_mode=False)

            if old_logits != None:
                diff = (logits - old_logits).abs()
                print(f"Debug -> {diff.mean()}")
                print("mean:", diff.mean().item())
                print("max: ", diff.max().item())
                print("norm:", diff.norm(dim=-1).mean().item())
            old_logits = logits

            sampling_logits = logits #/ temperature
            next_token_id = torch.distributions.Categorical(
                    logits=sampling_logits
                    ).sample().detach()

            #tokenized_words.append(next_token_id)

            print(next_token_id)

            next_word = tokenizer.decode(next_token_id[0].tolist())
            print("--------------------------------", next_word)
            print(tokenized_words)
            #list_of_words+=

            #next_embedding = next_embedding[:, None, :]  # [B, 1, D]
            #original = torch.cat([original, next_embedding], dim=1)
            break


        return tokenized_words

    def counterfactual_generate(self, sentence, temperature=1.0):
        tokenizer = get_encoder()
        tokenized_words = tokenizer.encode(sentence)
        print(tokenized_words)

        list_of_words = sentence.split(" ")

        # ----------------- original generation --------------------------------

        old_logits = None

        original = self.model.transformer.get_embeddings(torch.tensor(tokenized_words).to(device))
        w_ = len(tokenized_words)
        input_ = {"seqs": original.to(device), "original_mask": torch.tensor([w_*[1]+[0]]).to(device), "loss_mask": torch.tensor([w_*[0]+[1]]).to(device)}

        original_logits = self.diffusion_step(input_)

        sampling_logits = original_logits / temperature
        next_token_id = torch.distributions.Categorical(logits=sampling_logits).sample().detach().item()

        next_word = tokenizer.decode([next_token_id])

        next_words = ["paper", "plane", "cat", next_word]

        # ---------------------------------------------------------------------

        for word in next_words:

            tokenized_words = tokenizer.encode(sentence + word)
            
            original = self.model.transformer.get_embeddings(torch.tensor(tokenized_words).to(device))
            w_ = len(tokenized_words)
            print(w_)
            input_ = {"seqs": original.to(device), "original_mask": torch.tensor([w_*[1]+w_*[0]]).to(device), "loss_mask": torch.tensor([(w_*[0]+w_*[1])]).to(device)}

            logits = self.diffusion_step(input_)

            print(f"==========={word}=====================")

            diff = (logits - original_logits).abs()
            print(f"Debug -> {diff.mean()}")
            print("mean:", diff.mean().item())
            print("max: ", diff.max().item())
            print("norm:", diff.norm(dim=-1).mean().item())

            sampling_logits = logits / temperature
            next_token_id = torch.distributions.Categorical(
                    logits=sampling_logits
                    ).sample().detach().item()

            #tokenized_words.append(next_token_id)

            next_word = tokenizer.decode([next_token_id])
            print("--------------------------------", next_word)
            #list_of_words+=

            #next_embedding = next_embedding[:, None, :]  # [B, 1, D]
            #original = torch.cat([original, next_embedding], dim=1)


        return tokenized_words
