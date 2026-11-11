import h5py
import numpy as np
import polars as pl
import pyranges as pr
import pytest

from transcript_transformer.data import process_ribo_data
from transcript_transformer.processing import (
    construct_output_table,
    csv_to_gtf,
    process_seq_preds,
)
from transcript_transformer.util_functions import vec2DNA


@pytest.fixture
def predictions(assembly):
    with h5py.File(assembly, "a") as f:
        tr = f["transcript"]
        scores = tr.create_dataset(
            "tis_transformer_score", shape=tr["seq"].shape,
            dtype=h5py.vlen_dtype(np.dtype("float32")),
        )
        for i, name in enumerate(tr["transcript_id"].asstr()[:]):
            values = np.zeros(tr["transcript_len"][i], dtype=np.float32)
            if name in ["plus", "minus"]:
                values[6] = 0.9
                values[9] = 0.95  # Above threshold, but not a permitted start codon.
                values[-1] = 0.8  # Incomplete codon: not an ORF.
            elif name in ["noncoding", "nonstop"]:
                values[3] = 0.8
            scores[i] = values
    return assembly


@pytest.mark.parametrize("include_nonstop", [False, True])
def test_orf_results(predictions, include_nonstop):
    prefix = str(predictions.parent / "result")
    df, filtered, novel = construct_output_table(
        str(predictions), prefix, prob_cutoff=0.5, min_ORF_len=9,
        exclude_invalid_TTS=not include_nonstop, return_ORF_coords=True,
    )
    expected = {
        "plus": (7, 22, 7, 22, 15, "MKPGF", "annotated CDS"),
        "minus": (7, 22, 74, 49, 15, "MKPGF", "annotated CDS"),
        "noncoding": (4, 13, 4, 13, 9, "MPK", "lncRNA-ORF"),
    }
    if include_nonstop:
        expected["nonstop"] = (4, 0, 4, -1, 60, "M" + "K" * 19, "annotated CDS")
    columns = ["TIS_pos", "TTS_pos", "TIS_coord", "TTS_coord", "ORF_len", "protein_seq", "ORF_type"]
    for frame in [df, filtered]:
        actual = {row[0]: row[1:] for row in frame.select("transcript_id", *columns).rows()}
        assert actual == expected
    assert novel["transcript_id"].to_list() == ["noncoding"]
    assert df.filter(pl.col("transcript_id") == "minus")["ORF_coords"][0] == "73;74;50;62"

    # FASTA-only and assembly-based processing should agree on transcript-level ORFs.
    with h5py.File(predictions, "r") as f:
        tr = f["transcript"]
        fasta = process_seq_preds(
            tr["transcript_id"].asstr()[:], tr["tis_transformer_score"][:],
            [vec2DNA(seq) for seq in tr["seq"]], 0.5,
            min_ORF_len=9, include_invalid_TTS=include_nonstop,
        )
    assert {
        (r["transcript_id"], r["TIS_pos"], r["protein_sequence"])
        for r in fasta.iter_rows(named=True)
    } == {(name, values[0], values[5]) for name, values in expected.items()}

    for frame, suffix in [(df, ".redundant"), (filtered, ""), (novel, ".novel")]:
        csv = pl.read_csv(f"{prefix}{suffix}.csv")
        assert set(csv["ORF_id"]) == set(frame["ORF_id"])
        csv_to_gtf(str(predictions), frame, prefix + suffix, "test")
    gtf = pr.read_gtf(prefix + ".gtf", as_df=True)
    minus = gtf[(gtf.transcript_id == "minus_7") & (gtf.Feature == "CDS")]
    assert list(zip(minus.Start + 1, minus.End)) == [(73, 74), (50, 62)]
    start = gtf[(gtf.transcript_id == "minus_7") & (gtf.Feature == "start_codon")]
    assert list(zip(start.Start + 1, start.End)) == [(73, 74), (62, 62)]


@pytest.mark.parametrize("parallel", [False, True])
def test_grouped_ribosome_results(predictions, parallel):
    sam = str(predictions.parent / "features.sam")
    process_ribo_data(
        str(predictions), {"group": {"a": sam, "b": sam}}, parallel=parallel,
    )
    with h5py.File(predictions, "r") as f:
        tr = f["transcript"]
        output = [
            (b"group|" + name, scores)
            for name, scores in zip(tr["transcript_id"][:], tr["tis_transformer_score"][:])
        ]
    df, _, _ = construct_output_table(
        str(predictions), str(predictions.parent / "grouped"),
        prob_cutoff=0.5, min_ORF_len=9, correction=True,
        ribo_output=output, grouped_ribo_ids={"group": ["a", "b"]}, parallel=parallel,
    )
    actual = {
        row[0]: row[1:]
        for row in df.select(
            "transcript_id", "reads_in_transcript", "reads_in_ORF", "reads_in_frame_frac"
        ).rows()
    }
    assert actual == {"plus": (8, 8, 0.75), "minus": (4, 4, 1.0), "noncoding": (2, 2, 1.0)}


@pytest.mark.parametrize("ribo", [False, True])
def test_empty_output(predictions, ribo):
    prefix = str(predictions.parent / "empty")
    kwargs = {}
    if ribo:
        process_ribo_data(
            str(predictions), {"sample": {"sample": str(predictions.parent / "features.sam")}},
        )
        with h5py.File(predictions, "r") as f:
            tr = f["transcript"]
            kwargs = {
                "ribo_output": [
                    (b"sample|" + name, score)
                    for name, score in zip(tr["transcript_id"][:], tr["tis_transformer_score"][:])
                ],
                "grouped_ribo_ids": {"sample": ["sample"]},
                "correction": True,
            }
    frames = construct_output_table(str(predictions), prefix, prob_cutoff=1.0, **kwargs)
    for frame, suffix in zip(frames, [".redundant", "", ".novel"]):
        assert len(frame) == 0
        assert "ORF_id" in pl.read_csv(prefix + suffix + ".csv").columns
        csv_to_gtf(str(predictions), frame, prefix + suffix, "test")
        assert (predictions.parent / ("empty" + suffix + ".gtf")).read_text() == ""
