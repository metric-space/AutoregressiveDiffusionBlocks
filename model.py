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
