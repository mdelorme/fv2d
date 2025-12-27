"""
Regression test for `fv2d`. The aim of these tests is not to assess the quality of the result,
but rather to ensure that new features in the code do not break existing functionality of `fv2d.
Solutions are compared to reference data generated with a known working version of the code, which
has been validated at higher resolutions and compared to other codes. See the script `test_validation.py`
for validation tests.

If you wish to reproduce the reference data, see the script `run_all_ref.py` to (re)generate reference data.

Lucas Barbier-Goy - 28/12/2025.
"""

import subprocess
import sys
import numpy as np
from pathlib import Path

import pytest
from .ioutils import h5_to_numpy

REF_DIR = Path(__file__).parent / "refdata"
FV2D_EXE = Path(__file__).parent.parent / "build" / "fv2d"
FV2D_TMP_DIR  = Path(__file__).parent / "tmp_test_dir" # for improvement see : https://docs.pytest.org/en/stable/how-to/tmp_path.html
FV2D_TMP_DIR.mkdir(exist_ok=True)
TOLERANCE = 1e-2


def run_problem(inifile: str) -> None:
    with open("tmp_test_dir/run_log.txt", "a") as log_file:
        log_file.write(f"Running fv2d for {REF_DIR/inifile}...\n")
    subprocess.run([FV2D_EXE, REF_DIR / inifile], capture_output=False, cwd=FV2D_TMP_DIR)


def setup_test(pb_name: str, rsolver: str, run_type: str) -> None:
    filename = f"{pb_name}_{rsolver}" # Base filename w/o extension. Ex: sod_x_hll, sod_y_hllc...
    ref_path = REF_DIR / f"{filename}.npz"

    with np.load(ref_path) as ref_data:
        ref_density = ref_data['rho']
        ref_velocity_x = ref_data['u']
        ref_velocity_y = ref_data['v']
        ref_pressure = ref_data['prs']

    run_problem(f"{filename}.ini")
    h5_to_numpy(str(FV2D_TMP_DIR / f"{filename}.h5"), str(FV2D_TMP_DIR / f"{filename}.npz"))
    test_path = FV2D_TMP_DIR / f"{filename}.npz"

    with np.load(test_path) as test_data:
        test_density = test_data['rho']
        test_velocity_x = test_data['u']
        test_velocity_y = test_data['v']
        test_pressure = test_data['prs']

    assert np.allclose(test_density, ref_density, atol=TOLERANCE), "Density does not match reference data"
    assert np.allclose(test_velocity_x, ref_velocity_x, atol=TOLERANCE), "Velocity X does not match reference data"
    assert np.allclose(test_velocity_y, ref_velocity_y, atol=TOLERANCE), "Velocity Y does not match reference data"
    assert np.allclose(test_pressure, ref_pressure, atol=TOLERANCE), "Pressure does not match reference data"

# ======================================================================
# Sod shock tube test regression tests in x and y directions
# ======================================================================
# @pytest.mark.parametrize("config", (SOD_X_CONF,))
def test_sod_x_hll_regression() -> None:
    """Test the Sod shock tube with the HLL Riemann solver against reference data."""
    setup_test("sod_x", "hll", "hydro")

def test_sod_x_hllc_regression() -> None:
    """Test the Sod shock tube with the HLLC Riemann solver against reference data."""
    setup_test("sod_x", "hllc", "hydro")

def test_sod_x_fslp_regression() -> None:
    """Test the Sod shock tube with the FSLP Riemann solver against reference data."""
    setup_test("sod_x", "fslp", "hydro")

def test_sod_y_hll_regression() -> None:
    """Test the Sod shock tube in y direction with the HLL Riemann solver against reference data."""
    setup_test("sod_y", "hll", "hydro")

def test_sod_y_hllc_regression() -> None:
    """Test the Sod shock tube in y direction with the HLLC Riemann solver against reference data."""
    setup_test("sod_y", "hllc", "hydro")

def test_sod_y_fslp_regression() -> None:
    """Test the Sod shock tube in y direction with the FSLP Riemann solver against reference data."""
    setup_test("sod_y", "fslp", "hydro")

# ======================================================================
# Blast wave test regression tests
# ======================================================================

@pytest.mark.slow
def test_blast_hll_regression() -> None:
    """Test the blast wave with the HLL Riemann solver against reference data."""
    setup_test("blast", "hll", "hydro")

@pytest.mark.slow
def test_blast_hllc_regression() -> None:
    """Test the blast wave with the HLLC Riemann solver against reference data."""
    setup_test("blast", "hllc", "hydro")

@pytest.mark.slow
def test_blast_fslp_regression() -> None:
    """Test the blast wave with the FSLP Riemann solver against reference data."""
    setup_test("blast", "fslp", "hydro")

# =====================================================================
# Cattaneo (1991) convection regression tests
# =====================================================================

@pytest.mark.skip(reason="C91 data for HLL is not available at this time.")
def test_C91_hll_regression() -> None:
    """Test the Cattaneo convection test with the HLL Riemann solver against reference data."""
    setup_test("C91", "hll", "hydro")

@pytest.mark.skip(reason="C91 data for HLLC is not available at this time.")
def test_C91_hllc_regression() -> None:
    """Test the Cattaneo convection test with the HLLC Riemann solver against reference data."""
    setup_test("C91", "hllc", "hydro")

@pytest.mark.skip(reason="C91 data for FSLP is not available at this time.")
def test_C91_fslp_regression() -> None:
    """Test the Cattaneo convection test with the FSLP Riemann solver against reference data."""
    setup_test("C91", "fslp", "hydro")

# =====================================================================
# Diffusion test regression tests
# =====================================================================

@pytest.mark.slow
def test_diffusion_hll_regression() -> None:
    """Test the diffusion test with the HLL Riemann solver against reference data."""
    setup_test("diffusion", "hll", "hydro")

@pytest.mark.slow
def test_diffusion_hllc_regression() -> None:
    """Test the diffusion test with the HLLC Riemann solver against reference data."""
    setup_test("diffusion", "hllc", "hydro")

@pytest.mark.slow
def test_diffusion_fslp_regression() -> None:
    """Test the diffusion test with the FSLP Riemann solver against reference data."""
    setup_test("diffusion", "fslp", "hydro")

# =====================================================================
# Gresho Vortex regression tests
# =====================================================================

def test_gresho_vortex_hll_regression() -> None:
    """Test the Gresho vortex with the HLL Riemann solver against reference data."""
    setup_test("gresho_vortex", "hll", "hydro")

def test_gresho_vortex_hllc_regression() -> None:
    """Test the Gresho vortex with the HLLC Riemann solver against reference data."""
    setup_test("gresho_vortex", "hllc", "hydro")

def test_gresho_vortex_fslp_regression() -> None:
    """Test the Gresho vortex with the FSLP Riemann solver against reference data."""
    setup_test("gresho_vortex", "fslp", "hydro")

# =====================================================================
# H84 regression tests
# =====================================================================

def test_h84_hll_regression() -> None:
    """Test the H84 test with the HLL Riemann solver against reference data."""
    setup_test("H84", "hll", "hydro")

def test_h84_hllc_regression() -> None:
    """Test the H84 test with the HLLC Riemann solver against reference data."""
    setup_test("H84", "hllc", "hydro")

def test_h84_fslp_regression() -> None:
    """Test the H84 test with the FSLP Riemann solver against reference data."""
    setup_test("H84", "fslp", "hydro")

# =====================================================================
# Kelvin-Helmholtz regression tests
# =====================================================================
@pytest.mark.slow
def test_kelvin_helmholtz_hll_regression() -> None:
    """Test the Kelvin-Helmholtz with the HLL Riemann solver against reference data."""
    setup_test("kelvin_helmholtz", "hll", "hydro")

@pytest.mark.slow
def test_kelvin_helmholtz_hllc_regression() -> None:
    """Test the Kelvin-Helmholtz with the HLLC Riemann solver against reference data."""
    setup_test("kelvin_helmholtz", "hllc", "hydro")

@pytest.mark.skipif(sys.platform == "linux", reason="FSLP test is unstable on Linux CI runners.")
@pytest.mark.slow
def test_kelvin_helmholtz_fslp_regression() -> None:
    """Test the Kelvin-Helmholtz with the FSLP Riemann solver against reference data."""
    setup_test("kelvin_helmholtz", "fslp", "hydro")

# =====================================================================
# Rayleigh-Taylor instability regression tests
# =====================================================================

@pytest.mark.slow
def test_rayleigh_taylor_hll_regression() -> None:
    """Test the Rayleigh-Taylor instability with the HLL Riemann solver against reference data."""
    setup_test("rayleigh_taylor", "hll", "hydro")

@pytest.mark.slow
def test_rayleigh_taylor_hllc_regression() -> None:
    """Test the Rayleigh-Taylor instability with the HLLC Riemann solver against reference data."""
    setup_test("rayleigh_taylor", "hllc", "hydro")

@pytest.mark.slow
def test_rayleigh_taylor_fslp_regression() -> None:
    """Test the Rayleigh-Taylor instability with the FSLP Riemann solver against reference data."""
    setup_test("rayleigh_taylor", "fslp", "hydro")

# =====================================================================

# Clean up generated data after tests
# =====================================================================
def teardown_module() -> None:
    """Clean up temporary files after tests."""
    for file in FV2D_TMP_DIR.glob("*"):
        file.unlink()
    FV2D_TMP_DIR.rmdir()
