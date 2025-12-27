from config import get_test_config, load_config
from regression.sod import test_hll_sod_regression, test_hllc_sod_regression

def main():
    print("Test suite for fv2d")
    test_config = load_config()
    if test_config['hydro.sod_x']['active']:
        config = get_test_config('hydro.sod_x')
        test_hll_sod_regression(config.ref_path, config.sim_path, config.tolerance)
        test_hllc_sod_regression(config.ref_path, config.sim_path, config.tolerance)


if __name__ == "__main__":
    main()
