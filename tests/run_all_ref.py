"""
Module for running and generating all reference tests.
Lucas Barbier-Goy
28/12/2025
"""

import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from .ioutils import h5_to_numpy


FV2D_EXE = Path(__file__).parent.parent / "build" / "fv2d"
TMP_DIR  = Path(__file__).parent / "tmp_test_dir"
REF_DIR  = Path(__file__).parent / "refdata"
RSOLVERS = ["HLL", "HLLC", "FSLP"]

def get_files(inipath: Path) -> list[Path]:
    return list(inipath.glob('**/*.ini'))


def write_ini(filepath: Path, rsolver: str) -> None:
    data = filepath.open().read().replace("$rsolver$", rsolver.lower())
    tmp_dir = Path(__file__).parent / "tmp_test_dir"
    tmp_dir.mkdir(exist_ok=True)
    filename = f"{filepath.stem}_{rsolver}.ini"
    with open(tmp_dir / filename, "w") as f:
        f.write(data)


def write_all_inis(inifiles: list[Path], rsolvers: list[str]) -> None:
    for file in inifiles:
        for rsolv in rsolvers:
            write_ini(file, rsolv)


def run_problems(tmp_dir_path: Path) -> None:
    for inifile in tqdm(get_files(tmp_dir_path), desc="Generating reference data"):
        print(f"Running fv2d for {inifile.name}...")
        subprocess.run(["./fv2d", inifile], executable=FV2D_EXE, capture_output=True, cwd=REF_DIR)


def convert_h5_to_numpy(data_file_dir: Path) -> None:
    for h5file in tqdm(data_file_dir.glob("*.h5"), desc="Converting HDF5 to NumPy"):
        numpyfile = h5file.with_suffix('.npz')
        h5_to_numpy(str(h5file), str(numpyfile))


def cleanup() -> None:
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for h5file in REF_DIR.glob("*.h5"):
        h5file.unlink(missing_ok=False)
    for xmf_file in REF_DIR.glob("*.xmf"):
        xmf_file.unlink(missing_ok=False)


def main() -> None:
    """
    1. Get reference ini files for problems to run.
    2. For each ini file, create versions for each Riemann solver.
    3. Run fv2d on each ini file to generate reference data.
    4. Convert last iteration of each HDF5 file to NumPy format.
    5. Clean up temporary files.
    """
    files = get_files(REF_DIR)
    write_all_inis(files, rsolvers=RSOLVERS)
    run_problems(TMP_DIR)
    print("All reference data generated. Extracting last iterations to numpy format...")
    convert_h5_to_numpy(REF_DIR)
    print("All reference data generated. Cleaning up temporary files...")
    cleanup()


if __name__ == "__main__":
    main()
