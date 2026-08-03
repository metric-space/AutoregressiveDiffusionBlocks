import copy
import math
import os
import random
import struct
import zlib
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from gpt2 import GPT2LMHeadModel, GPT2Config
from dblock_modules import get_block_sigmas, get_discrete_sigmas
from data import get_encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## TODO: really should not make this a global man wtf
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
            get_discrete_sigmas(num_steps=4, dblock=True).to(
                self.device
            ),
        )
        self.save_hyperparameters(args)
        self.alternative = args.alternative

        self.transition = args.get("transition", None)
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
        self._attention_viz_count = 0

    def on_fit_start(self):
        self.logger.watch(
                self,
                log="gradients",
                log_freq=10,
                log_graph=False,
        )
        if self._attention_viz_enabled():
            root_dir = getattr(
                self,
                "attention_visualization_root_dir",
                self.trainer.default_root_dir,
            )
            self._attention_viz_dir = os.path.join(
                root_dir,
                self.args.attention_visualization.get("output_dir", "attention_maps"),
            )
            os.makedirs(self._attention_viz_dir, exist_ok=True)


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
        if self.alternative is not None and block_idx in self.alternative.keys():
            return self.alternative[block_idx]
        return block_idx

    
    def denoise(self, x, zt, sigma, block_idx=None, inference_mode=False):
        seq_device = x["seqs"].device if isinstance(x, dict) else x.device
        sigma = sigma.to(seq_device)
        zt = zt.to(seq_device)
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
                sigma_stuff=(c_out, zt, c_skip),
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

    def _attention_viz_enabled(self):
        cfg = self.args.get("attention_visualization", None)
        return bool(cfg is not None and cfg.get("enabled", False))

    def _should_visualize_attention(self, step):
        if not self._attention_viz_enabled() or step != "train":
            return False
        if getattr(self.trainer, "global_rank", 0) != 0:
            return False
        cfg = self.args.attention_visualization
        max_batches = cfg.get("max_batches", None)
        if max_batches is not None and self._attention_viz_count >= max_batches:
            return False
        every_n_steps = int(cfg.get("every_n_steps", 100))
        return every_n_steps > 0 and self.global_step % every_n_steps == 0

    @staticmethod
    def _attention_head_image(attention):
        attention = attention.float()
        attention = attention - attention.min()
        scale = attention.max().clamp_min(1e-8)
        attention = (attention / scale * 255).byte().numpy()

        red = attention
        green = (attention.astype(np.float32) * 0.55).astype(np.uint8)
        blue = 255 - attention
        rgb = np.stack([red, green, blue], axis=-1)
        return np.repeat(np.repeat(rgb, 12, axis=0), 12, axis=1)

    @staticmethod
    def _png_chunk(chunk_type, data):
        chunk = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + chunk
            + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        )

    @classmethod
    def _write_png(cls, path, image):
        height, width, _ = image.shape
        raw = b"".join(
            b"\x00" + np.ascontiguousarray(image[row]).tobytes()
            for row in range(height)
        )
        with open(path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
            handle.write(
                cls._png_chunk(
                    b"IHDR",
                    struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
                )
            )
            handle.write(cls._png_chunk(b"IDAT", zlib.compress(raw)))
            handle.write(cls._png_chunk(b"IEND", b""))

    def _save_attention_layer_image(self, weights, layer_idx, step_dir, sample_idx, max_heads):
        sample = weights[sample_idx]
        num_heads = min(sample.shape[0], max_heads)
        head_images = [self._attention_head_image(sample[head_idx]) for head_idx in range(num_heads)]
        tile_h, tile_w, _ = head_images[0].shape
        pad = 4
        columns = min(4, num_heads)
        rows = math.ceil(num_heads / columns)
        canvas = np.full(
            (
                rows * tile_h + (rows + 1) * pad,
                columns * tile_w + (columns + 1) * pad,
                3,
            ),
            255,
            dtype=np.uint8,
        )
        for head_idx, image in enumerate(head_images):
            x = pad + (head_idx % columns) * (tile_w + pad)
            y = pad + (head_idx // columns) * (tile_h + pad)
            canvas[y : y + tile_h, x : x + tile_w] = image
        path = os.path.join(step_dir, f"layer_{layer_idx:02d}.png")
        self._write_png(path, canvas)
        return path

    def _log_attention_maps(self, block_idx):
        maps = getattr(self.model, "last_attention_maps", [])
        if not maps:
            return

        cfg = self.args.attention_visualization
        save_local = bool(cfg.get("save_local", True))
        log_to_wandb = bool(cfg.get("log_to_wandb", True))
        if not save_local and not log_to_wandb:
            return

        sample_idx = int(cfg.get("sample_idx", 0))
        max_heads = int(cfg.get("max_heads", 12))
        max_layers = cfg.get("max_layers", None)
        if max_layers is not None:
            maps = maps[: int(max_layers)]

        step_dir = os.path.join(self._attention_viz_dir, f"step_{self.global_step:08d}")
        os.makedirs(step_dir, exist_ok=True)

        logged_images = {}
        for item in maps:
            weights = item["weights"]
            if sample_idx >= weights.shape[0]:
                continue
            if save_local:
                torch.save(weights, os.path.join(step_dir, f"layer_{item['layer']:02d}.pt"))
            path = self._save_attention_layer_image(
                weights=weights,
                layer_idx=item["layer"],
                step_dir=step_dir,
                sample_idx=sample_idx,
                max_heads=max_heads,
            )
            if log_to_wandb:
                logged_images[f"attention/block_{block_idx}/layer_{item['layer']}"] = path

        if log_to_wandb and logged_images:
            try:
                import wandb

                self.logger.experiment.log(
                    {
                        key: wandb.Image(path)
                        for key, path in logged_images.items()
                    },
                    step=int(self.global_step),
                )
            except Exception as exc:
                print(f"Attention visualization wandb logging failed: {exc}")

        if not save_local:
            for path in logged_images.values():
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.rmdir(step_dir)
            except OSError:
                pass

        self._attention_viz_count += 1


    def generate_inputs(self, original_seq, sigma=0.0, device="cpu"):

        original = self.model.transformer.get_embeddings(torch.tensor(original_seq, device=device))
        # NOTE: important bit
        original = self.model.transformer.add_position_embeddings(original)

        noised = torch.tensor(original_seq[1:], device=device)
        label  = torch.cat([torch.tensor(original_seq, device=device),torch.clone(noised)],dim=0)
        noised = self.model.transformer.get_embeddings(noised)

        noise = sigma * torch.randn_like(noised)
        # NOTE: important bit
        noised = noise + self.model.transformer.add_position_embeddings(noised, offset=1)

        input_ = torch.cat([original, noised], axis=0)
        noise  = torch.cat([torch.zeros(len(original_seq), noised.shape[-1], device=device), noise], dim=0)

        return {
                "label":    label,    # [1,seq]
                "noise":    noise,    # [1, seq, hidden_dim]
                "input":    input_,   # [1, seq, hidden_dim]
                "original": original, # for debugging otherwise useless
                "noised":   noised,    # fo debugging, otherwise useless
                }

    def generate_input_masks(self, seq, noised, device="cpu"):
        # make masks here

        seq_length = seq.shape[0]
        noised_length = noised.shape[0]

        # doubles as noise_mask
        loss_mask = torch.ones(seq_length + noised_length, device=device)
        loss_mask[:seq_length] = 0

        original_mask = torch.zeros(seq_length + noised_length, device=device)
        original_mask[:seq_length] = 1

        return {"original_mask" : original_mask,
                "loss_mask"     : loss_mask}


    def collate(self, list_of_dicts):
        collated_processed_batch = {}
        for key_ in list_of_dicts[0].keys(): # come back here and reduce keys
            v_ = torch.nn.utils.rnn.pad_sequence([kv[key_] for kv in list_of_dicts], batch_first=True)
            print(f"{key_ } {v_.shape}")
            #assert v_[0].ndim == 3
            collated_processed_batch[key_] = v_

        return collated_processed_batch


    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        model_device = next(self.model.parameters()).device

        batch = 10*batch

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


        sigmas = self.get_sigmas(self.current_training_block, len(batch)).to(
            model_device
        )

        zt = []

        # TODO: zip maybee better
        #
        for index, original_seq in enumerate(batch):
            # embedding

             print(f"Input: {text_decoder.decode(original_seq)}")

             inputs = self.generate_inputs(original_seq, sigma=sigmas[index], device=model_device)

             # NOTE: maybe we don't need this line
             zt.append(inputs["noise"])

             masks = self.generate_input_masks(inputs["original"], inputs["noised"], device=model_device)

             processed_batch.append(masks | inputs)

        block_idx = self.estimate_target_layer(sigmas)

        print(f"Processed batch: {processed_batch}")
        
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

        # ---------- experimental ----------------------------------------

        collated_processed_batch = self.collate(processed_batch)
        

        labels = collated_processed_batch["label"]
        del collated_processed_batch["label"]

        zt = collated_processed_batch["noise"]
        del collated_processed_batch["noise"]

        # rename keys here
        collated_processed_batch["seqs"] = collated_processed_batch.pop("input")

        capture_attention = self._should_visualize_attention(step)
        self.model.set_attention_capture(capture_attention)
        try:
            logits = self.denoise(collated_processed_batch, zt, sigmas, block_idx, inference_mode=False)
        finally:
            self.model.set_attention_capture(False)
        if capture_attention:
            self._log_attention_maps(block_idx)

        # ================ Debugging process =============================

        # NOTE: the mask flattens

        print("Warning! batch dimension is toast here, come back here to correct after testing phase is over")

        logits = logits #[collated_processed_batch["loss_mask"].bool()] # should be [seq, hidden_dim]

        print(logits.shape)

        predicted_token_ids = torch.argmax(logits, dim=-1)

        #print(f"DEBUG: what is being mapped: {sigmas}")
        #for i in range(len(predicted_token_ids)):
        #    print(i, text_decoder.decode([predicted_token_ids[i].item()]), "->", text_decoder.decode([labels[0,i].item()]))

        
        #for i in range(predicted_token_ids.shape[0]):
        #    a_ = predicted_token_ids[i]
        #    print(f"Predicted tokens : {text_decoder.decode(a_.tolist())}")
        #    t_ = temp_mask[0]
        #    print(f"Predicted tokens part2 : {text_decoder.decode(a_[t_].tolist())}")
        #    print(f"Expected tokens : {text_decoder.decode(labels[i].tolist())}")


        loss = F.cross_entropy(
                logits.view(-1, 50257), labels.view(-1),
                reduction='none'
        )

        print("loss shape is: ", loss.shape)
        #loss = loss.reshape(-1, loss.size()[-1])
        ce_loss = loss.mean()
        w = self.get_weights(sigmas)[:, None]
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
        scheduler = instantiate(
            self.args.scheduler,
            optimizer=optimizer,
            T_max=self.trainer.estimated_stepping_batches,
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
        x_device = x.device
        #x = self.model.transformer.add_position_embeddings(x)
        bsz = x.shape[0]
        slen = original["loss_mask"].sum().item()
        hidden_size = self.model.config.n_embd
        z = torch.randn(
            bsz,
            (slen - 1),
            hidden_size,
            device=x_device,
            dtype=x.dtype,
        )
        #z = self.model.transformer.add_position_embeddings(z, offset = 1)
        z *= torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x.new_ones([bsz])

        #z = z[:,None,:]
        print("-------------------------------------------------------------")

        for i in range(self.sigmas.shape[0] - 1):
            print(self.sigmas[i])
            original["seqs"] = torch.cat([x, z], dim=1)
            original["seqs"] = self.model.transformer.add_position_embeddings(original["seqs"])
            sigma = self.sigmas[i] * s_in
            next_sigma = self.sigmas[i + 1] * s_in
            # denoise
            logits = self.denoise(original, torch.cat([torch.zeros_like(x),z],dim=1), sigma, inference_mode=inference_mode)[:, -(slen - 1):,:]

            probs = torch.softmax(logits, dim=-1)
            denoised = probs @ self.model.transformer.wte.weight
            #denoised = probs

            print("z:", z.shape)
            print("logits:", logits.shape)
            print("probs:", probs.shape)
            print("denoised:", denoised.shape)
            print("argmax:", torch.argmax(logits, dim=-1))
            # to d
            d = (z - denoised) / sigma #[:, None] # none is for the hidden_dim
            dt = next_sigma - sigma
            print(f"dt*d is {dt*d}")
            # euler step
            euler_step = z + dt*d#[:, None] * d # None is for the hidden dim
            z = euler_step

        min_sigma = self.sigmas[-1].item()
        original["seqs"] = torch.cat([x, z], dim=1) 
        original["seqs"] = self.model.transformer.add_position_embeddings(original["seqs"])
        sigmas = torch.full((x.shape[0],), min_sigma, device=x_device, dtype=x.dtype)
        logits = self.denoise(original,  torch.cat([torch.zeros_like(x),z],dim=1), sigmas, inference_mode=inference_mode)[:, -(slen - 1):, :]
        return logits


    def generate(self, sentence, num_new_tokens, temperature=1.0):
        tokenizer = get_encoder()
        model_device = next(self.model.parameters()).device
        tokenized_words = tokenizer.encode(sentence)
        #tokenized_words = [50256]
        print(tokenized_words)

        list_of_words = sentence.split(" ")

        old_logits = None

        for _ in range(num_new_tokens):

            original = torch.tensor(tokenized_words, device=model_device)
            original = self.model.transformer.get_embeddings(original)
            w_ = len(tokenized_words)
            input_ = {
                "seqs": original[None,...],
                "original_mask": torch.tensor([w_*[1]+[0]], device=original.device),
                "loss_mask": torch.tensor([[1]+(w_-1)*[0]+[1]], device=original.device),
            }

            logits = self.diffusion_step(input_, inference_mode=True)

            if old_logits != None:
                diff = (logits - old_logits).abs()
                print(f"Debug -> {diff.mean()}")
                print("mean:", diff.mean().item())
                print("max: ", diff.max().item())
                print("norm:", diff.norm(dim=-1).mean().item())
            old_logits = logits

            next_token_id = torch.argmax(logits, dim=-1).item()
            print(f"next_token_id: {next_token_id}")

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
        model_device = next(self.model.parameters()).device
        tokenized_words = tokenizer.encode(sentence)
        #tokenized_words = [50256]
        print(tokenized_words)

        list_of_words = sentence.split(" ")

        old_logits = None

        for _ in range(num_new_tokens):

            original = torch.tensor(tokenized_words, device=model_device)
            original = self.model.transformer.get_embeddings(original)
            w_ = len(tokenized_words)
            print(f"Length of tokenized inference input is : {w_}")
            input_ = {
                "seqs": original[None,...],
                "original_mask": torch.tensor(
                    [(w_ + num_new_tokens )*[1]+(w_ - 1 + num_new_tokens)*[0]],
                    device=original.device,
                ),
                "loss_mask": torch.tensor(
                    [(w_ + num_new_tokens)*[0]+ (w_ - 1 + num_new_tokens)*[1]],
                    device=original.device,
                ),
            }

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

    def sigma_sweep(self, sentence):

        sigmas = torch.arange(1.0, 80.0, 2.5)

        tokenizer = get_encoder()
        model_device = next(self.model.parameters()).device
        tokenized_words = tokenizer.encode(sentence)
        #tokenized_words = [50256]
        print(tokenized_words)

        list_of_words = sentence.split(" ")
        original = torch.tensor(tokenized_words, device=model_device)

        # NOTE: hard code for now
        block_idx = 3

        

        for sigma in sigmas:

             inputs = self.generate_inputs(original, sigma=sigma, device=model_device)
             masks = self.generate_input_masks(inputs["original"], inputs["noised"], device=model_device)

             processed_batch = [(masks | inputs)]

             collated_processed_batch = self.collate(processed_batch)

             labels = collated_processed_batch["label"]
             del collated_processed_batch["label"]

             zt = collated_processed_batch["noise"]
             del collated_processed_batch["noise"]

             # rename keys here
             collated_processed_batch["seqs"] = collated_processed_batch.pop("input")
             logits = self.denoise(collated_processed_batch, zt, torch.tensor([sigma]), block_idx, inference_mode=False)

             # ================ Debugging process =============================
             # NOTE: the mask flattens

             logits = logits[collated_processed_batch["loss_mask"].bool()] # should be [seq, hidden_dim]
             predicted_token_ids = torch.argmax(logits, dim=-1)
             # decoded = 
             print(f"For sigma={sigma} : {predicted_token_ids.tolist()}  {tokenizer.decode(predicted_token_ids.tolist())}\n\n")
            


    def counterfactual_generate(self, sentence, temperature=1.0):
        tokenizer = get_encoder()
        model_device = next(self.model.parameters()).device
        tokenized_words = tokenizer.encode(sentence)
        print(tokenized_words)

        list_of_words = sentence.split(" ")

        # ----------------- original generation --------------------------------

        old_logits = None

        original = self.model.transformer.get_embeddings(
            torch.tensor(tokenized_words, device=model_device)
        )
        w_ = len(tokenized_words)
        input_ = {
            "seqs": original,
            "original_mask": torch.tensor([w_*[1]+[0]], device=original.device),
            "loss_mask": torch.tensor([w_*[0]+[1]], device=original.device),
        }

        original_logits = self.diffusion_step(input_)

        sampling_logits = original_logits / temperature
        next_token_id = torch.distributions.Categorical(logits=sampling_logits).sample().detach().item()

        next_word = tokenizer.decode([next_token_id])

        next_words = ["paper", "plane", "cat", next_word]

        # ---------------------------------------------------------------------

        for word in next_words:

            tokenized_words = tokenizer.encode(sentence + word)
            
            original = self.model.transformer.get_embeddings(
                torch.tensor(tokenized_words, device=model_device)
            )
            w_ = len(tokenized_words)
            print(w_)
            input_ = {
                "seqs": original,
                "original_mask": torch.tensor([w_*[1]+w_*[0]], device=original.device),
                "loss_mask": torch.tensor([(w_*[0]+w_*[1])], device=original.device),
            }

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
