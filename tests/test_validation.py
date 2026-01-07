"""
Validation tests for `fv2d`. The aim of these tests is to assess the quality of results provided by `fv2d`.
Solutions are compared to reference data generated with known working codes and setups.

Lucas Barbier-Goy - 28/12/2025.
"""

from math import e
import subprocess
import numpy as np
from pathlib import Path
from .ioutils import h5_to_numpy

REF_DIR = Path(__file__).parent / "validation_data"
FV2D_EXE = Path(__file__).parent.parent / "build" / "fv2d"
FV2D_TMP_DIR  = Path(__file__).parent / "tmp_test_dir"
FV2D_TMP_DIR.mkdir(exist_ok=True)
TOLERANCE = 1e-8

def test_test():
    assert True

def teardown_module() -> None:
    """Clean up temporary files after tests."""
    for file in FV2D_TMP_DIR.glob("*"):
        file.unlink()
    if FV2D_TMP_DIR.exists():
        FV2D_TMP_DIR.rmdir()
