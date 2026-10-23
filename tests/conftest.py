import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

KESTREL = os.path.join(ROOT, "scenarios", "kestrel")
MODELS = os.path.join(ROOT, "tests", "models")


@pytest.fixture(scope="session")
def kestrel_model():
    from plyxis.lang import compile_model
    return compile_model([os.path.join(KESTREL, "field.plx")], physical=True)


@pytest.fixture(scope="session")
def small_model():
    from plyxis.lang import compile_model
    return compile_model([os.path.join(MODELS, "small.plx")], include_stdlib=False, physical=True)
