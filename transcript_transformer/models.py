import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as tm
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding



# ==========================================
# 1. Positional Embeddings
# ==========================================


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("emb", emb)

    def forward(self, x):
        return self.emb[None, : x.shape[1], :].to(x)


# ==========================================
# 2. Hybrid Flash Attention Block
#    (Splits heads between Local & Global)
# ==========================================


class HybridFlashBlock(nn.Module):
    def __init__(
        self, dim, heads, dim_head, local_attn_heads=None, window_size=256, dropout=0.1
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size

        # Default: If not specified, make all heads local except 2 (unless heads < 2)
        if local_attn_heads is None:
            self.local_attn_heads = max(0, heads - 2)
        else:
            self.local_attn_heads = local_attn_heads

        self.global_heads = heads - self.local_attn_heads
        inner_dim = heads * dim_head

        # Architecture sanity checks
        assert (
            self.local_attn_heads + self.global_heads == heads
        ), "Local + Global heads must equal total heads"

        self.norm1 = nn.LayerNorm(dim)

        # Fused projection for efficiency
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None, rotary_emb=None, causal=False):
        """
        x: (Batch, Seq, Dim)
        mask: (Batch, Seq) bool tensor, True for active tokens.
        rotary_emb: Pre-computed rotary table
        """
        B, N, D = x.shape
        window = self.window_size

        # 1. Pad Sequence for Windowing
        pad_len = (window - (N % window)) % window
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
            mask_padded = F.pad(mask, (0, pad_len), value=False) if mask is not None else None
        else:
            x_padded = x
            mask_padded = mask

        N_pad = x_padded.shape[1]

        # 2. Pre-Norm & Projection
        normed_x = self.norm1(x_padded)
        qkv = self.to_qkv(normed_x).chunk(3, dim=-1)

        # Reshape to (B, Heads, N_pad, Dim_Head)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # 3. Apply RoPE Globally
        if rotary_emb is not None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        # 4. Split Heads
        if self.local_attn_heads > 0 and self.global_heads > 0:
            q_l, q_g = q.split([self.local_attn_heads, self.global_heads], dim=1)
            k_l, k_g = k.split([self.local_attn_heads, self.global_heads], dim=1)
            v_l, v_g = v.split([self.local_attn_heads, self.global_heads], dim=1)
        elif self.local_attn_heads > 0:
            q_l, k_l, v_l = q, k, v
            q_g = k_g = v_g = None
        else:
            q_g, k_g, v_g = q, k, v
            q_l = k_l = v_l = None

        outputs = []

        # --- BRANCH A: Local Attention (O(N)) ---
        if q_l is not None:
            # Reshape: (B, H, N_pad, D) -> (B * Num_Windows, H, Window_Size, D)
            q_l = rearrange(q_l, "b h (nw w) d -> (b nw) h w d", w=window)
            k_l = rearrange(k_l, "b h (nw w) d -> (b nw) h w d", w=window)
            v_l = rearrange(v_l, "b h (nw w) d -> (b nw) h w d", w=window)

            # Local Mask: (B, N_pad) -> (B * Num_Windows, 1, 1, Window_Size)
            attn_mask_l = None
            if mask_padded is not None:
                attn_mask_l = rearrange(mask_padded, "b (nw w) -> (b nw) 1 1 w", w=window)

            # Flash Attention on Windows with fallback
            with torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                ]
            ):
                out_l = F.scaled_dot_product_attention(
                    q_l,
                    k_l,
                    v_l,
                    attn_mask=attn_mask_l,
                    dropout_p=0.1 if self.training else 0.0,
                    is_causal=False,  # Local windows are dense
                )

            # Reshape Back
            out_l = rearrange(out_l, "(b nw) h w d -> b h (nw w) d", b=B)
            outputs.append(out_l)

        # --- BRANCH B: Global Attention (O(N^2)) ---
        if q_g is not None:
            # Global Mask: (B, N_pad) -> (B, 1, 1, N_pad) or (B, 1, N_pad, N_pad) if causal
            attn_mask_g = None
            is_sdpa_causal = False
            
            if mask_padded is not None:
                if causal:
                    # SDPA doesn't like both attn_mask and is_causal=True
                    # We create a combined causal + padding mask
                    # (N_pad, N_pad) causal
                    c_mask = torch.tril(torch.ones(N_pad, N_pad, device=x.device, dtype=torch.bool))
                    # (B, 1, 1, N_pad) padding
                    p_mask = mask_padded.unsqueeze(1).unsqueeze(2)
                    attn_mask_g = c_mask & p_mask
                else:
                    attn_mask_g = mask_padded.unsqueeze(1).unsqueeze(2)
            else:
                is_sdpa_causal = causal

            with torch.nn.attention.sdpa_kernel(
                [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                ]
            ):
                out_g = F.scaled_dot_product_attention(
                    q_g,
                    k_g,
                    v_g,
                    attn_mask=attn_mask_g,
                    dropout_p=0.1 if self.training else 0.0,
                    is_causal=is_sdpa_causal,
                )
            outputs.append(out_g)

        # 5. Concatenate Results
        if len(outputs) == 2:
            out = torch.cat(outputs, dim=1)
        else:
            out = outputs[0]

        # 6. Output Projection
        out = rearrange(out, "b h n d -> b n (h d)")

        # 7. Remove Padding & Residual Connection
        if pad_len > 0:
            out = out[:, :N, :]

        x = x + self.to_out(out)

        # 8. Feed Forward
        x = x + self.ff(self.norm2(x))

        return x


# ==========================================
# 3. Main Model Class
# ==========================================


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
        dim_head,
        causal,
        emb_dropout,
        ff_dropout,
        attn_dropout,
        local_attn_heads,
        local_window_size,
        mlm,
        mask_frac,
        rand_frac,
        metrics,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Flash Attention Guardrails ---
        if dim % heads != 0:
            raise ValueError(f"Dim ({dim}) must be divisible by heads ({heads}).")

        self.head_dim = dim // heads

        # Head dim must be divisible by 8 for Flash Attention kernels
        if self.head_dim % 8 != 0:
            # Fallback suggestion if user provided incompatible params
            suggested_dim = heads * (round(self.head_dim / 8) * 8)
            print(
                f"WARNING: head_dim ({self.head_dim}) is not a multiple of 8. Flash Attention may fall back to slow math."
            )
            print(f"Suggestion: Change dim to {suggested_dim}")

        # --- Model Components ---
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        # Instantiate Hybrid Layers
        self.layers = nn.ModuleList(
            [
                HybridFlashBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=self.head_dim,
                    local_attn_heads=local_attn_heads,
                    window_size=local_window_size,
                    dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )

        # --- Task Specific Setup (MLM / Classification) ---
        if mlm in ["ribo", "seq"]:
            self.mlm = True
            self.hparams.mask_c = mask_frac
            self.hparams.mask_m = self.hparams.mask_c + (1 - self.hparams.mask_c) * (
                1 - rand_frac
            )
            self.loss_fn = (
                nn.BCEWithLogitsLoss() if mlm == "ribo" else nn.CrossEntropyLoss()
            )
            pos_label = 21 if mlm == "ribo" else num_tokens
            if mlm == "ribo":
                self.ribo_mlm_emb = nn.Embedding(1, dim)
            else:
                self.mask_token = 4
        else:
            self.mlm = False
            self.loss_fn = nn.CrossEntropyLoss()
            pos_label = 2
            if "ROC" in metrics:
                self.val_rocauc = tm.AUROC(task="binary")
            if "PR" in metrics:
                self.val_prauc = tm.AveragePrecision(task="binary")

        # --- Embeddings ---
        self.ff_1 = nn.Linear(dim, dim * 2)
        self.ff_2 = nn.Linear(dim * 2, pos_label)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(emb_dropout)

        if use_ribo:
            self.ff_emb_1 = nn.Linear(1, dim)
            self.ff_emb_2 = nn.Linear(dim, 6 * dim)
            self.ff_emb_3 = nn.Linear(6 * dim, dim)
            self.scalar_emb = nn.Sequential(
                self.ff_emb_1,
                self.activation,
                self.ff_emb_2,
                self.activation,
                self.ff_emb_3,
                nn.Tanh(),
            )
            self.ribo_count_emb = nn.Embedding(1, dim)
            self.ribo_read_emb = nn.Embedding(21, dim)

        if use_seq:
            self.nuc_emb = nn.Embedding(num_tokens, dim)

        # Fixed absolute sinusoidal pos embedding
        self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len + 512)

    def forward(self, batch, eval=False):
        # 1. Masking Logic
        y_mask = batch["y"] != -1
        x_mask = torch.clone(y_mask)
        x_mask[:, 0] = 1
        x_mask[torch.arange(x_mask.shape[0]), x_mask.sum(dim=1)] = 1

        # 2. Embedding Logic
        x = self.parse_embeddings(batch)

        if self.mlm:
            dist = torch.empty(batch["y"].shape, device=self.device).uniform_(0, 1)
            y_mask = torch.logical_and(dist > self.hparams.mask_c, y_mask)
            if "seq" in batch.keys():
                y_true = batch["seq"][torch.logical_and(x_mask, y_mask)]
                x = self.rand_seq(x, dist, eval)
            else:
                y_true = (batch["ribo"][torch.logical_and(x_mask, y_mask)] > 0).float()
                x = self.rand_ribo(x, dist, eval)
        else:
            y_true = batch["y"][y_mask].view(-1)

        # 3. Add Absolute Position & Dropout
        x = self.dropout(x + self.pos_emb(x))

        # 4. Transformer Layers (Hybrid)
        for layer in self.layers:
            x = layer(
                x, mask=x_mask, rotary_emb=self.rotary_emb, causal=self.hparams.causal
            )

        # 5. Classification Head
        x = x[torch.logical_and(x_mask, y_mask)]
        x = x.view(-1, self.hparams.dim)
        x = self.activation(self.ff_1(x))
        x = self.ff_2(x)

        return x.float(), y_true, y_mask

    # --- Helper Methods (Preserved from your original) ---

    def parse_embeddings(self, batch):
        xs = []
        if "ribo" in batch.keys():
            # counts per position
            ribo_data = batch["ribo"].float()
            if ribo_data.shape[-1] == 1:
                counts = ribo_data
            else:
                counts = ribo_data.sum(dim=-1).unsqueeze(-1)
            
            # Embed count scalar
            x_counts = self.scalar_emb(counts) * self.ribo_count_emb.weight
            xs.append(x_counts)

            if ribo_data.shape[-1] > 1:
                # read fraction per position for multi-channel data
                reads_sum = ribo_data.sum(axis=-1).unsqueeze(-1)
                x_frac = torch.nan_to_num(torch.div(ribo_data, reads_sum))
                # linear combination between read length fraction and read length embedding
                x_read_len = torch.einsum("ikj,jl->ikl", [x_frac, self.ribo_read_emb.weight])
                xs.append(x_read_len)
                
        if "seq" in batch.keys():
            xs.append(self.nuc_emb(batch["seq"]))

        return torch.sum(torch.stack(xs), dim=0)

    def rand_seq(self, x, dist, eval):
        mask = dist <= self.hparams.mask_m
        rand = torch.logical_and(dist > self.hparams.mask_c, mask)
        mask_token = torch.full(x.shape[:2], self.mask_token, device=self.device)

        if not eval:
            x[mask] = self.nuc_emb(mask_token)[mask]
            # Random token injection
            rand_tokens = torch.randint(0, 4, x.shape[:2], device=self.device)
            x[rand] = self.nuc_emb(rand_tokens)[rand]
        else:
            x[mask] = self.nuc_emb(mask_token)[mask]
        return x

    def rand_ribo(self, x, dist, eval):
        # Logic for Ribo MLM masking
        mask = dist <= self.hparams.mask_m
        rand = torch.logical_and(dist > self.hparams.mask_c, mask)

        if not eval:
            x[mask] = 0  # Zero out masked
            # Add random noise for 'rand' fraction
            noise = torch.randn_like(x[rand])
            x[rand] = noise
        else:
            x[mask] = 0
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.decay_rate
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        y_hat, y_true, y_mask = self(batch)
        loss = self.loss_fn(y_hat, y_true)
        self.log("train_loss", loss, sync_dist=True, batch_size=len(y_true))
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y_true, y_mask = self(batch, eval=True)
        loss = self.loss_fn(y_hat, y_true)
        self.log("val_loss", loss, sync_dist=True, batch_size=len(y_true))

        if not self.mlm:
            probs = torch.softmax(y_hat, dim=1)[:, 1]
            if hasattr(self, "val_rocauc"):
                self.val_rocauc(probs, y_true)
            if hasattr(self, "val_prauc"):
                self.val_prauc(probs, y_true)
        return loss

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
