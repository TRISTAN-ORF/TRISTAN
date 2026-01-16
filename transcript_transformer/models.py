import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as tm

from torch.nn.modules.activation import ReLU

torch.serialization.add_safe_globals([ReLU])

import math

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=30002):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Handle odd d_model
        pe[:, 0:d_model:2] = torch.sin(position * div_term[:(d_model+1)//2])
        if d_model > 1:
            pe[:, 1:d_model:2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TranscriptSeqRiboEmb(pl.LightningModule):
    def __init__(
        self,
        use_seq,
        use_ribo,
        num_tokens,
        lr,
        decay_rate,
        warmup_steps,
        max_seq_len,
        dim,
        depth,
        heads,
        causal,
        emb_dropout,
        ff_dropout,
        attn_dropout,
        mlm,
        mask_frac,
        rand_frac,
        metrics,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Ensure d_model allows for valid head_dim for Flash Attention (>= 8 typically better)
        # Bumping to 16 to better match the capacity of the original Performer implementation
        head_dim = dim // heads
        if head_dim < 16:
            self.d_model = heads * 16
        else:
            self.d_model = dim
            
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=heads,
                dim_feedforward=dim * 4, # Maintain FFN size based on input dim
                dropout=attn_dropout,
                activation="relu",
                batch_first=True,
            ) for _ in range(depth)
        ])
        
        if self.d_model != dim:
            self.proj_in = torch.nn.Linear(dim, self.d_model)
            self.proj_out = torch.nn.Linear(self.d_model, dim)
        else:
            self.proj_in = None
            self.proj_out = None

        if mlm in ["ribo", "seq"]:
            self.mlm = True
            self.hparams.mask_c = mask_frac
            self.hparams.mask_m = self.hparams.mask_c + (1 - self.hparams.mask_c) * (
                1 - rand_frac
            )
            if mlm == "ribo":
                self.loss = torch.nn.BCEWithLogitsLoss()
                pos_label = 21
                self.ribo_mlm_emb = torch.nn.Embedding(1, dim)
            else:
                self.loss = torch.nn.CrossEntropyLoss()
                pos_label = num_tokens
                self.mask_token = 4  # token 4 is represents N
        else:
            self.mlm = False
            self.loss = torch.nn.CrossEntropyLoss()
            pos_label = 2
            if "ROC" in metrics:
                self.val_rocauc = tm.AUROC(task="binary")
                self.test_rocauc = tm.AUROC(task="binary")

            if "PR" in metrics:
                self.val_prauc = tm.AveragePrecision(task="binary")
                self.test_prauc = tm.AveragePrecision(task="binary")

        self.ff_1 = torch.nn.Linear(dim, dim * 2)
        self.ff_2 = torch.nn.Linear(dim * 2, pos_label)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(emb_dropout)

        if use_ribo:
            self.ff_emb_1 = torch.nn.Linear(1, dim)
            self.ff_emb_2 = torch.nn.Linear(dim, 6 * dim)
            self.ff_emb_3 = torch.nn.Linear(6 * dim, dim)
            self.tanh = torch.nn.Tanh()
            self.scalar_emb = torch.nn.Sequential(
                self.ff_emb_1,
                self.relu,
                self.ff_emb_2,
                self.relu,
                self.ff_emb_3,
                self.tanh,
            )
            self.ribo_count_emb = torch.nn.Embedding(1, dim)
            self.ribo_read_emb = torch.nn.Embedding(21, dim)
        if use_seq:
            self.nuc_emb = torch.nn.Embedding(num_tokens, dim)

        self.pos_emb = SinusoidalPositionalEncoding(dim, max_seq_len + 2)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["mlm"] = self.mlm

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        if not self.mlm and "mlm" in checkpoint.keys() and checkpoint["mlm"]:
            for key in ["ff_2.weight", "ff_2.bias", "ff_1.weight", "ff_1.bias"]:
                state_dict.pop(key)
            checkpoint["mlm"] = False
        if "pos_emb.emb" in state_dict:
            if self.pos_emb.weight.shape != state_dict["pos_emb.emb"].shape:
                state_dict.pop("pos_emb.emb")
                state_dict.pop("layer_pos_emb.emb")
        elif self.pos_emb.weight.shape != state_dict["pos_emb.weight"].shape:
            state_dict.pop("pos_emb.weight")
            # Remove layer_pos_emb.weight if present in old checkpoint (it was there in Performer)
            state_dict.pop("layer_pos_emb.weight", None)
        checkpoint["state_dict"] = state_dict

    def parse_embeddings(self, batch):
        xs = []
        if "ribo" in batch.keys():
            # if using offsets that map to a single position
            if batch["ribo"].shape[-1] == 1:
                xs.append(self.scalar_emb(batch["ribo"]) * self.ribo_count_emb.weight)
            else:
                # counts per position
                counts = batch["ribo"].sum(dim=-1).unsqueeze(-1)  # bs, l, 1x
                xs.append(self.scalar_emb(counts) * self.ribo_count_emb.weight)
                # read fraction per position
                x = torch.nan_to_num(
                    torch.div(batch["ribo"], batch["ribo"].sum(axis=-1).unsqueeze(-1))
                )
                # linear combination between read length fraction and read length embedding
                xs.append(torch.einsum("ikj,jl->ikl", [x, self.ribo_read_emb.weight]))
        if "seq" in batch.keys():
            xs.append(self.nuc_emb(batch["seq"]))

        return torch.sum(torch.stack(xs), dim=0)

    def rand_seq(self, x, dist, eval=False):
        # random masking of tokens
        mask = torch.logical_and(dist > self.hparams.mask_c, dist < self.hparams.mask_m)
        x[mask] = self.nuc_emb.weight[torch.full((mask.sum(),), self.mask_token)]
        # randomizing of tokens
        if not eval:
            mask = dist > self.hparams.mask_m
            idx_vec = torch.randint(
                0, self.mask_token, (mask.sum(),), device=self.device
            )
            x[mask] = self.nuc_emb.weight[idx_vec]

        return x

    def rand_ribo(self, x, dist, eval=False):
        # random masking of tokens
        mask = torch.logical_and(dist > self.hparams.mask_c, dist < self.hparams.mask_m)
        x[mask] = self.ribo_mlm_emb.weight[torch.full((mask.sum(),), 0)]
        # randomizing of read fraction columns within batch
        if not eval:
            mask = dist > self.hparams.mask_m
            x[mask] = x[mask][torch.randperm(mask.sum())]

        return x

    def forward(self, batch, eval=False):
        y_mask = batch["y"] != -1
        # include start/end transcript token to be part of x_mask
        x_mask = torch.clone(y_mask)
        x_mask[:, 0] = 1
        x_mask[torch.arange(x_mask.shape[0]), x_mask.sum(dim=1)] = 1

        x = self.parse_embeddings(batch)
        if self.mlm:
            dist = torch.empty(batch["y"].shape, device=self.device).uniform_(
                0,
                1,
            )
            y_mask = torch.logical_and(dist > self.hparams.mask_c, y_mask)
            if "seq" in batch.keys():
                y_true = batch["seq"][torch.logical_and(x_mask, y_mask)]
                x = self.rand_seq(x, dist, eval)
            else:
                y_true = (batch["ribo"][torch.logical_and(x_mask, y_mask)] > 0).type(
                    torch.float
                )
                x = self.rand_ribo(x, dist, eval)
        else:
            y_true = batch["y"][y_mask].view(-1)

        x = self.pos_emb(x)
        x = self.dropout(x)

        if self.proj_in is not None:
            x = self.proj_in(x)
            
        x = x.contiguous() # Ensure contiguity for kernels

        # Use modern sdpa_kernel context manager
        # Force Flash Attention (disable math and mem_efficient to test compatibility)
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
            # Manual loop over layers
            with torch.autocast(device_type='cuda', dtype=torch.float16): # or bfloat16
                for layer in self.layers:
                    x = layer(x, src_key_padding_mask=~x_mask.bool())

        
        # Projection back from (B, L, d_model) to (B, L, dim).
        if self.proj_out is not None:
            x = self.proj_out(x)

        # Select valid tokens
        x = x[torch.logical_and(x_mask, y_mask)]
        x = x.view(-1, self.hparams.dim)

        # Feed forward
        x = F.relu(self.ff_1(x))
        x = self.ff_2(x)

        return x.float(), y_true, y_mask

    def training_step(self, batch, batch_idx):
        y_hat, y_true, _ = self(batch)

        loss = self.loss(y_hat, y_true)
        self.log("train_loss", loss, batch_size=len(y_true))

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y_true, _ = self(batch)

        self.log("val_loss", self.loss(y_hat, y_true), batch_size=len(y_true))
        if hasattr(self, "val_prauc"):
            self.val_prauc(F.softmax(y_hat, dim=1)[:, 1], y_true)
            self.log(
                "val_prauc",
                self.val_prauc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y_true),
            )
        if hasattr(self, "val_rocauc"):
            self.val_rocauc(F.softmax(y_hat, dim=1)[:, 1], y_true)
            self.log(
                "val_rocauc",
                self.val_rocauc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y_true),
            )

    def test_step(
        self,
        batch,
        batch_idx,
    ):
        y_hat, y_true, _ = self(batch)

        self.log("test_loss", self.loss(y_hat, y_true), batch_size=len(y_true))
        if hasattr(self, "test_prauc"):
            self.test_prauc(F.softmax(y_hat, dim=1)[:, 1], y_true)
            self.log(
                "test_prauc",
                self.test_prauc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y_true),
            )
        if hasattr(self, "test_rocauc"):
            self.test_rocauc(F.softmax(y_hat, dim=1)[:, 1], y_true)
            self.log(
                "test_rocauc",
                self.test_rocauc,
                on_step=False,
                on_epoch=True,
                batch_size=len(y_true),
            )

    def predict_step(self, batch, batch_idx):
        y_hat, y_true, y_mask = self(batch)

        if hasattr(self, "test_prauc"):
            self.test_prauc(F.softmax(y_hat, dim=1)[:, 1], y_true)
        if hasattr(self, "test_rocauc"):
            self.test_rocauc(F.softmax(y_hat, dim=1)[:, 1], y_true)

        splits = torch.cumsum(y_mask.sum(dim=1), 0, dtype=torch.long).cpu()
        split_probs = torch.tensor_split(F.softmax(y_hat, dim=1)[:, 1], splits)[:-1]
        probs_grouped = [t.cpu().numpy() for t in split_probs]
        split_trues = torch.tensor_split(batch["y"][y_mask], splits)[:-1]
        if self.mlm:
            trues_grouped = [t.cpu().numpy() for t in split_trues]
        else:
            trues_grouped = [t.cpu().numpy().astype(bool) for t in split_trues]

        return probs_grouped, trues_grouped, batch["x_id"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        def lambda1(epoch):
            return self.hparams.decay_rate

        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambda1
        )

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        optimizer.step(closure=optimizer_closure)
