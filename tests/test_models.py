import json
from importlib.resources import files
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from transcript_transformer.models import TranscriptSeqRiboEmb
from transcript_transformer.transcript_loader import collate_fn


@pytest.mark.parametrize("name,checkpoint", [
    ("human", "tt_models/Homo_sapiens.GRCh38.113_f0.tt.ckpt"),
    ("mouse", "tt_models/Mus_musculus.GRCm39.112_f0.tt.ckpt"),
    ("ribotie", "rt_models/50perc_06_23_f0.rt.ckpt"),
])
def test_published_model_predictions(name, checkpoint):
    data = Path(__file__).parent / "data"
    reference = json.loads((data / "model_scores.json").read_text())[name]
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        ribo = name == "ribotie"
        model = TranscriptSeqRiboEmb.load_from_checkpoint(
            files("transcript_transformer.pretrained").joinpath(checkpoint),
            map_location="cpu", metrics=[], max_seq_len=512, strict=False,
            use_seq=not ribo, use_ribo=ribo, mlm=False, mask_frac=0.85, rand_frac=0.15,
        )
        model.eval()
        with h5py.File(data / "tt_predicting.h5", "r") as f:
            sequence = f["transcript/seq"][0][:128]
        features = {"seq": sequence}
        if ribo:
            reads = np.zeros((128, 21), dtype=np.float32)
            reads[::3, 8] = 1
            features = {"ribo": reads}
        batch = collate_fn([
            [["example"], [features], [np.zeros(128, dtype=np.int64)]]
        ])
        with torch.no_grad():
            scores = torch.softmax(model(batch)[0], dim=1)[:, 1].numpy()
        assert scores.shape == (128,)
        assert np.isfinite(scores).all()
        np.testing.assert_allclose(
            scores[reference["positions"]], reference["scores"], rtol=1e-4, atol=1e-7,
        )
    finally:
        torch.set_num_threads(threads)
