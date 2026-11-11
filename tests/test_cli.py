import os
import pytest
from itertools import product
import subprocess
import h5py
import numpy as np
import polars as pl
from pathlib import Path
import shutil
import sys
import yaml


@pytest.fixture(scope="module")
def cli_workspace(tmp_path_factory):
    source = Path(__file__).parent
    workspace = tmp_path_factory.mktemp("cli")
    shutil.copytree(source / "data", workspace / "data")
    shutil.copytree(source / "configs", workspace / "configs")
    (workspace / "dbs").mkdir()
    (workspace / "out").mkdir()
    for path in (workspace / "configs").glob("*.yml"):
        config = yaml.safe_load(path.read_text())
        config.update(
            accelerator="cpu", devices=1, num_workers=0, cores=2,
            max_seq_len=1000, max_memory=5000, metrics=[],
            cond=None,
        )
        if path.stem.startswith("rt_"):
            # Length-limited fixtures can otherwise lose every sample in a fold.
            config["cond"] = {"ribo": {"num_reads": "x >= 0"}}
        if path.stem in ["tt_training", "rt_pretraining"]:
            # Training completion must not depend on randomly initialized scores;
            # fixed-prediction tests check the scientific output separately.
            config.update(
                dim=12, depth=1, heads=2, local_attn_heads=1,
                dim_head=8, nb_features=8, local_window_size=16,
                prob_cutoff=1.0,
            )
        path.write_text(yaml.safe_dump(config))
    return workspace


@pytest.fixture(autouse=True)
def setup_module(monkeypatch, cli_workspace):
    monkeypatch.chdir(cli_workspace)
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")


def remove_if_exists(file_paths):
    """Remove the file if it exists."""
    for file_path in file_paths:
        if Path(file_path).exists():
            os.remove(file_path)


def assert_if_exists(file_paths):
    """Assert that the file exists."""
    for file_path in file_paths:
        assert Path(file_path).exists(), f"{file_path} was not created"


def test_cli_version():
    result = subprocess.run(["ribotie", "--version"], capture_output=True, text=True)

    # Check that the return code is zero (indicating success)
    assert result.returncode == 0

    result = subprocess.run(
        ["tis_transformer", "--version"], capture_output=True, text=True
    )

    # Check that the return code is zero (indicating success)
    assert result.returncode == 0


def test_cli_help():
    result = subprocess.run(["ribotie", "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "ribotie" in result.stdout
    # Add more checks based on expected help output


# TIS Transformer
def test_sequence_data_loading():
    remove_if_exists(["dbs/tt_loading.h5"])

    command = [
        "tis_transformer",
        "configs/tt_loading.yml",
    ]

    result = subprocess.run(command, check=True, text=True)
    file_path = Path("dbs/tt_loading.h5")
    assert file_path.exists(), "HDF5 file was not created"

    f = h5py.File(file_path, "r")
    assert "transcript" in f.keys(), "HDF5 file does not contain 'transcript' group"
    assert len(f["transcript"].keys()) > 3, "HDF5 file does not contain expected keys"
    assert len(f["transcript/seqname"]) > 0, "HDF5 file does not contain 'seqname' data"


def test_sequence_training():
    files = [
        f"out/tt_training.{ext}"
        for ext in ["npy", "csv", "gtf", "redundant.csv", "redundant.gtf"]
    ]
    remove_if_exists(files)

    command = [
        "tis_transformer",
        "configs/tt_training.yml",
    ]

    subprocess.run(command, check=True, text=True)

    assert_if_exists(files)


def test_sequence_predicting():
    files = [
        f"out/tt_predicting.{ext}"
        for ext in ["npy", "csv", "gtf", "redundant.csv", "redundant.gtf"]
    ]
    remove_if_exists(files)

    command = [
        "tis_transformer",
        "configs/tt_predicting.yml",
        "data/tt_params.yml",
    ]

    subprocess.run(command, check=True, text=True)

    assert_if_exists(files)


def test_fasta_predicting():
    subprocess.run(
        [
            "tis_transformer", "--model", "human", "--fasta", "data/features.fa",
            "--out_prefix", "out/fasta", "--accelerator", "cpu", "--num_workers", "0",
            "--max_seq_len", "1000", "--prob_cutoff", "0", "--min_ORF_len", "9",
            "--include_invalid_TTS",
        ],
        check=True, text=True,
    )
    predictions = np.load("out/fasta.npy", allow_pickle=True)
    assert len(predictions) == 5
    result = pl.read_csv("out/fasta.csv")
    plus = result.filter(pl.col("transcript_id") == "plus")
    assert plus["TIS_pos"].to_list() == [7]
    assert plus["protein_sequence"].to_list() == ["MKPGF"]


# RiboTIE
def test_rt_sam_data_loading():
    remove_if_exists(["dbs/rt_sam_loading.h5"])

    command = [
        "ribotie",
        "configs/rt_sam_loading.yml",
    ]

    result = subprocess.run(command, check=True, text=True)
    file_path = Path("dbs/rt_sam_loading.h5")
    assert file_path.exists(), "HDF5 file was not created"
    f = h5py.File(file_path, "r")["transcript"]
    assert "riboseq" in f.keys(), "HDF5 file does not contain 'ribo' group"
    assert (
        len(f["riboseq/sample_1/5"].keys()) > 3
    ), "HDF5 file does not contain expected keys"
    assert (
        len(f["riboseq/sample_1/5/data"]) > 0
    ), "HDF5 file does not contain 'data' entries"


def test_rt_bam_data_loading():
    remove_if_exists(["dbs/rt_bam_loading.h5"])

    command = [
        "ribotie",
        "configs/rt_bam_loading.yml",
    ]

    subprocess.run(command, check=True, text=True)
    file_path = Path("dbs/rt_bam_loading.h5")
    assert file_path.exists(), "HDF5 file was not created"
    f = h5py.File(file_path, "r")["transcript"]
    assert "riboseq" in f.keys(), "HDF5 file does not contain 'ribo' group"
    assert (
        len(f["riboseq/sample_1/5"].keys()) > 3
    ), "HDF5 file does not contain expected keys"
    assert (
        len(f["riboseq/sample_1/5/data"]) > 0
    ), "HDF5 file does not contain 'data' entries"


def test_rt_pretraining():
    files = [
        f"out/rt_pretraining_pretrain_f{i}.rt.{ext}"
        for i, ext in list(product(range(2), ["ckpt"]))
    ] + [
        f"out/rt_pretraining_pretrain.{ext}"
        for ext in ["csv", "gtf", "redundant.csv", "redundant.gtf"]
    ]

    remove_if_exists(files)

    command = [
        "ribotie",
        "configs/rt_pretraining.yml",
    ]
    subprocess.run(command, check=True, text=True)

    assert_if_exists(files)


def test_rt_training():
    files = [
        f"out/rt_training_sample_{i}.{ext}"
        for i, ext in list(product(range(1, 4), ["npy", "csv", "gtf"]))
    ] + [
        f"out/rt_training_sample_{i}.redundant.{ext}"
        for i, ext in list(product(range(1, 4), ["csv", "gtf"]))
    ]
    remove_if_exists(files)

    command = [
        "ribotie",
        "configs/rt_training.yml",
    ]
    subprocess.run(command, check=True, text=True)

    assert_if_exists(files)
