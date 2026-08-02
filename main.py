import os
import datetime
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from data import load_data, TextData
from model import TransformerBlockModel
from gpt2 import normalize_transition_schedule

import hydra
from omegaconf import DictConfig

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args: DictConfig) -> None:
    L.seed_everything(args.seed)

    transition = normalize_transition_schedule(args.get("transition", None))
    if transition is not None:
        args.num_epochs = sum(epochs for _, epochs in transition)

    data = TextData() #load_data(args)
    #args.image_size = data.image_size
    #args.num_labels = data.num_labels
    # model = load_model(args)

    model = TransformerBlockModel(args)
    if args.ckpt_path is not None:
        nowname = os.path.basename(os.path.dirname(args.ckpt_path))
    else:
        now = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9), name="JST")
        ).strftime("%Y-%m-%dT%H-%M-%S")
        nowname = now + f"-{args.model_type}" + args.postfix
        if nowname.startswith("_"):
            nowname = nowname[1:]
    print("Experiment Name:", nowname)
    logdir = os.path.join("logs", nowname)
    logger = WandbLogger(
        project="diffblocks-tt",
        name=f"split-and-merge-44444",
        version=nowname,
        offline=args.debug,
        save_dir=logdir,
        # group=f"{args.data_name}",
    )
    max_epochs = args.num_epochs
    if args.model_type == "dblock" and transition is None:
        # Align total iterations across the full network because one dblock step
        # corresponds to one block.
        max_epochs *= args.num_blocks

    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=args.save_every_n_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=logdir,
                monitor= None, #"val/acc" if data.val_key is not None else None,
                mode="max",
                save_top_k=args.save_top_k,
                save_on_train_epoch_end=True,
                every_n_epochs=None, #args.save_every_n_epochs if data.val_key is None else None,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        strategy=DDPStrategy(find_unused_parameters=args.model_type == "dblock")
        if args.devices > 1
        else "auto",
        devices=args.devices,
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=4,
        # precision="bf16-mixed",
    )
    if args.stage == "train":
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
        #if data.test_key is not None:
        #    trainer.test(model, data.test_dataloader(), ckpt_path="best")
        # model.sigma_sweep("pizza with fox is the best") #, 5)
        model.generate("pizza with", 5)
    else:
        assert args.ckpt_path is not None
        trainer.test(model, data, ckpt_path=args.ckpt_path)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()
