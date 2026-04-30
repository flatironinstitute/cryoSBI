from pathlib import Path

import matplotlib

# Run matplotlib in headless mode so plot tests don't try to open a display
# in CI / SSH sessions and so figures don't leak across tests.
matplotlib.use("Agg")

TESTS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    """No-op hook kept for future per-test markers (e.g. gpu)."""
    return
