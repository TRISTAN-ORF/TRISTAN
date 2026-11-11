import h5py
import numpy as np
import polars as pl
import pyfaidx
import pyranges as pr
import pysam
import pytest
from h5max import load_sparse_matrix

from transcript_transformer import COMPAT_MAPPING
from transcript_transformer.data import (
    parse_transcriptome,
    process_ribo_data,
    save_transcriptome_to_h5,
)
from transcript_transformer.util_functions import vec2DNA, vec2prot


def test_transcript_content(assembly):
    with h5py.File(assembly, "r") as f:
        tr = f["transcript"]
        ids = tr["transcript_id"].asstr()[:].tolist()
        plus, minus, noncoding = [ids.index(name) for name in ["plus", "minus", "noncoding"]]
        sequence = "CCCCCCATGAAACCCGGGTTTTAA" + "C" * 36
        assert vec2DNA(tr["seq"][plus]) == sequence
        assert vec2DNA(tr["seq"][minus]) == sequence
        assert vec2prot(tr["canonical_protein_seq"][plus]) == "MKPGF"
        assert vec2prot(tr["canonical_protein_seq"][minus]) == "MKPGF"
        assert tr["tag"].asstr()[plus] == "basic,Ensembl_canonical"
        assert tr["transcript_biotype"].asstr()[plus] == "protein_coding"
        assert tr["gene_name"].asstr()[plus] == "Plus"
        np.testing.assert_array_equal(tr["exon_coords"][minus], [73, 80, 11, 62])
        np.testing.assert_array_equal(tr["exon_idxs"][minus], [0, 8, 8, 60])
        assert tr["canonical_TIS_coord"][minus] == 74
        assert tr["canonical_TTS_coord"][minus] == 49
        assert vec2DNA(tr["seq"][noncoding]).startswith("NNNATGCCCAAATAA")
        assert len(tr["CDS_coords"][noncoding]) == 0
        assert len(tr["canonical_protein_seq"][noncoding]) == 0
        for i in range(len(ids)):
            assert len(tr["seq"][i]) == len(tr["tis"][i]) == tr["transcript_len"][i]
            assert set(tr["seq"][i]) <= {0, 1, 2, 3, 4}


@pytest.mark.parametrize("ids", [["plus", "minus"], ["noncoding"]])
def test_transcript_subsets(assembly, ids):
    # Uniform-length and single-transcript inputs must retain the same HDF5 layout.
    gtf = pl.from_pandas(
        pr.read_gtf(str(assembly.parent / "features.gtf"), as_df=True, duplicate_attr=True)
    ).rename(COMPAT_MAPPING)
    gtf = gtf.filter(pl.col("transcript_id").is_in(ids)).with_columns(
        pl.col("start") + 1, pl.col("exon_number").cast(pl.Int32, strict=False)
    )
    with pyfaidx.Fasta(str(assembly.parent / "features.fa")) as fasta:
        db = parse_transcriptome(gtf, fasta)
    with h5py.File(assembly.parent / "subset.h5", "w") as f:
        save_transcriptome_to_h5(f, db)
        with h5py.File(assembly, "r") as original:
            original_ids = original["transcript/transcript_id"].asstr()[:].tolist()
            for i, name in enumerate(f["transcript/transcript_id"].asstr()[:]):
                for key in ["seq", "tis", "exon_idxs", "exon_coords", "CDS_coords", "canonical_protein_seq"]:
                    assert f[f"transcript/{key}"].shape == (len(ids),)
                    np.testing.assert_array_equal(
                        f[f"transcript/{key}"][i],
                        original[f"transcript/{key}"][original_ids.index(name)],
                    )


@pytest.mark.parametrize("format", ["sam", "bam"])
@pytest.mark.parametrize("parallel", [False, True])
def test_read_counts(assembly, format, parallel):
    sam = assembly.parent / "features.sam"
    path = sam
    if format == "bam":
        path = sam.with_suffix(".bam")
        pysam.sort("-o", str(path), str(sam))
    process_ribo_data(str(assembly), {"sample": {"sample": str(path)}}, parallel=parallel)
    counts_path = assembly.with_name("assembly_sample.h5") if parallel else assembly
    with h5py.File(assembly, "r") as f:
        ids = f["transcript/transcript_id"].asstr()[:].tolist()
    with h5py.File(counts_path, "r") as f:
        group = f["transcript/riboseq/sample/5"]
        expected = {"plus": 4, "minus": 2, "noncoding": 1, "nonstop": 1, "silent": 0}
        assert dict(zip(ids, group["num_reads"][:])) == expected
        for i, name in enumerate(ids):
            matrix = load_sparse_matrix(group, i, format="csr")
            assert matrix.sum() == expected[name]
            if name == "plus":
                assert matrix[0, 6] == 2  # Duplicate 20-nt reads at the same position.
                assert matrix[8, 9] == 1
                assert matrix[20, 20] == 1  # 40 nt included; 19 and 41 nt excluded.
