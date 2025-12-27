"""
Module defining configurations for running the `fv2d` test suite with `pytest`
Lucas Barbier-Goy
28/12/2025
"""

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
