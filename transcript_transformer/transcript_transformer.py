import os
import numpy as np
import logging
import itertools

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .models import TranscriptSeqRiboEmb
from .transcript_loader import (
    h5pyDataModule,
    DNADatasetBatches,
    collate_fn,
)
from .util_functions import DNA2vec, parse_fasta

from pdb import set_trace


def device_info_filter(record):
    return "PU available: " not in record.getMessage()


def acc_info_filter(record):
    return "LOCAL_RANK: " not in record.getMessage()


logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(device_info_filter)
logging.getLogger("pytorch_lightning.accelerators.cuda").addFilter(acc_info_filter)


def train(args, test_model=True, enable_model_summary=True):
    if args.transfer_checkpoint or (
        hasattr(args, "checkpoint_data") and args.checkpoint_data
    ):
        if hasattr(args, "checkpoint_data") and args.checkpoint_data:
            # Load from dictionary
            model = TranscriptSeqRiboEmb(
                args.use_seq,
                args.use_ribo,
                args.num_tokens,
                args.lr,
                args.decay_rate,
                args.warmup_steps,
                args.max_seq_len,
                args.dim,
                args.depth,
                args.heads,
                args.dim_head,
                False,
                args.emb_dropout,
                args.ff_dropout,
                args.attn_dropout,
                args.local_attn_heads,
                args.local_window_size,
                args.mlm,
                args.mask_frac,
                args.rand_frac,
                args.metrics,
                args.scheduler,
            )
            model.load_state_dict(args.checkpoint_data["state_dict"])
        else:
            model = TranscriptSeqRiboEmb.load_from_checkpoint(
                args.transfer_checkpoint,
                strict=False,
                use_seq=args.use_seq,
                use_ribo=args.use_ribo,
                lr=args.lr,
                decay_rate=args.decay_rate,
                warmup_steps=args.warmup_steps,
                max_seq_len=args.max_seq_len,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                dim_head=args.dim_head,
                local_attn_heads=args.local_attn_heads,
                local_window_size=args.local_window_size,
                mlm=args.mlm,
                mask_frac=args.mask_frac,
                rand_frac=args.rand_frac,
            )
    else:
        model = TranscriptSeqRiboEmb(
            args.use_seq,
            args.use_ribo,
            args.num_tokens,
            args.lr,
            args.decay_rate,
            args.warmup_steps,
            args.max_seq_len,
            args.dim,
            args.depth,
            args.heads,
            args.dim_head,
            False,
            args.emb_dropout,
            args.ff_dropout,
            args.attn_dropout,
            args.local_attn_heads,
            args.local_window_size,
            args.mlm,
            args.mask_frac,
            args.rand_frac,
            args.metrics,
            args.scheduler,
        )
    tr_loader = h5pyDataModule(
        args.h5_path,
        args.exp_path,
        args.y_path,
        args.id_path,
        args.seqn_path,
        args.use_seq,
        args.grouped_ribo_ids,
        args.offsets,
        train=args.train,
        val=args.val,
        test=args.test,
        strict_validation=args.strict_validation,
        max_memory=args.max_memory,
        max_transcripts_per_batch=args.max_transcripts_per_batch,
        num_workers=args.num_workers,
        cond=args.cond,
        leaky_frac=args.leaky_frac,
        collate_fn=collate_fn,
        parallel=args.parallel,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch:02d}_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    log_dir = os.path.join(os.path.dirname(args.out_prefix), "models")
    tb_logger = pl.loggers.TensorBoardLogger(
        ".", os.path.join(log_dir, os.path.basename(args.out_prefix))
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        enable_model_summary=enable_model_summary,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
        ],
        logger=tb_logger if not args.debug else False,
        enable_checkpointing=not args.debug,
    )

    
    # Calculate and log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\t -- Number of learnable parameters: {trainable_params:,}")

    trainer.fit(model, datamodule=tr_loader)
    if test_model:
        print(trainer.checkpoint_callbacks)
        trainer.test(model, datamodule=tr_loader, ckpt_path="best")

    return trainer, model


def predict(args, trainer=None, model=None):
    if args.accelerator == "cpu":
        map_location = torch.device("cpu")
    else:
        map_location = torch.device("cuda")

    if trainer is None:
        trainer = pl.Trainer(
            accelerator=args.accelerator,
            strategy=args.strategy,
            devices=args.devices,
            precision=args.precision,
            enable_checkpointing=False,
            logger=None,
        )

    if model is None:
        if hasattr(args, "checkpoint_data") and args.checkpoint_data:
            model = TranscriptSeqRiboEmb(
                args.use_seq,
                args.use_ribo,
                args.num_tokens,
                args.lr,
                args.decay_rate,
                args.warmup_steps,
                args.max_seq_len,
                args.dim,
                args.depth,
                args.heads,
                args.dim_head,
                False,
                args.emb_dropout,
                args.ff_dropout,
                args.attn_dropout,
                args.local_attn_heads,
                args.local_window_size,
                args.mlm,
                args.mask_frac,
                args.rand_frac,
                args.metrics,
                args.scheduler,
            )
            model.load_state_dict(args.checkpoint_data["state_dict"])
            model.to(map_location)
        else:
            model = TranscriptSeqRiboEmb.load_from_checkpoint(
                args.transfer_checkpoint,
                map_location=map_location,
                strict=False,
                max_seq_len=args.max_seq_len,
                dim=args.dim,
                depth=args.depth,
                heads=args.heads,
                dim_head=args.dim_head,
                local_attn_heads=args.local_attn_heads,
                local_window_size=args.local_window_size,
                mlm=False,
                mask_frac=0.85,
                rand_frac=0.15,
                metrics=[],
            )
        ckpt_path = None
    else:
        ckpt_path = "best"

    if not hasattr(args, "fasta") or args.fasta is None:
        tr_loader = h5pyDataModule(
            args.h5_path,
            args.exp_path,
            args.y_path,
            args.id_path,
            args.seqn_path,
            args.use_seq,
            args.grouped_ribo_ids,
            args.offsets,
            train=args.train,
            val=args.val,
            test=args.test,
            max_memory=args.max_memory,
            max_transcripts_per_batch=args.max_transcripts_per_batch,
            num_workers=args.num_workers,
            cond=args.cond,
            collate_fn=collate_fn,
            parallel=args.parallel,
        )
    else:
        tr_ids, tr_seqs = parse_fasta(args.fasta, args.max_seq_len)
        assert len(tr_seqs) > 0, "no valid sequences in fasta"
        x_data = [DNA2vec(seq) for seq in tr_seqs]
        tr_loader = DataLoader(
            DNADatasetBatches(tr_ids, x_data), collate_fn=collate_fn, batch_size=1
        )

    out = trainer.predict(model, dataloaders=tr_loader, ckpt_path=ckpt_path)
    if out is not None:
        ids = list(itertools.chain(*[o[2] for o in out]))
        preds = list(itertools.chain(*[o[0] for o in out]))

        if hasattr(args, "fasta") and (args.fasta is not None):
            targets = list(itertools.chain(*[o[1] for o in out]))
            out = [ids, [p.astype(np.float32) for p in preds], targets]
        else:
            out = [ids, [p.astype(np.float32) for p in preds]]
    else:
        out = []

    np.save(
        f"{args.out_prefix}.npy",
        np.array(out, dtype=object).T,
    )
    print(f"\t -- Raw model outputs saved to '{args.out_prefix}.npy'")

    return


def main():
    print(
        "transcript_transformer script is now deprecated, Use 'tis_transformer' or 'ribotie' instead."
    )


if __name__ == "__main__":
    main()
