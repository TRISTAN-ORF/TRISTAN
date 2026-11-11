import shutil
from pathlib import Path

import pytest

from transcript_transformer.data import process_seq_data


DATA = Path(__file__).parent / "data"


@pytest.fixture
def assembly(tmp_path):
    for name in ["features.fa", "features.gtf", "features.sam"]:
        shutil.copyfile(DATA / name, tmp_path / name)
    h5_path = tmp_path / "assembly.h5"
    process_seq_data(
        str(h5_path),
        str(tmp_path / "features.gtf"),
        str(tmp_path / "features.fa"),
        backup_path=None,
        backup=False,
    )
    return h5_path
