from configparser import ConfigParser
from dataclasses import dataclass


def load_config(path: str = 'config.ini') -> ConfigParser:
    """Load the configuration file for regression tests."""
    config = ConfigParser()
    config.read(path)
    return config

@dataclass
class TestConfig:
    active: bool
    ref_path: str
    sim_path: str
    riemann: list[str]
    tolerance: float
    regression: bool
    validation: bool
    unit: bool

def get_test_config(test_name: str, path: str = 'config.ini') -> TestConfig:
    """Retrieve the configuration for a specific test."""
    config = load_config(path)
    test_cfg = config[test_name]
    return TestConfig(
        active=test_cfg.getboolean('active', False),
        ref_path=test_cfg['ref_path'],
        sim_path=test_cfg.get('sim_path', 'output/'),
        riemann=[solver.strip() for solver in test_cfg['riemann'].strip('[]').split(',')],
        tolerance=test_cfg.getfloat('tolerance', 1e-6),
        regression=test_cfg.getboolean('regression', False),
        validation=test_cfg.getboolean('validation', False),
        unit=test_cfg.getboolean('unit', False)
    )
