import sys

from flfm.settings import settings
from flfm.tests.conftest import clear_all_imports, reset_settings  # noqa: F401

BACKENDS = ("jax", "torch")


def test_import(clear_all_imports, reset_settings):  # noqa: F811
    # Test that default backend hasn't been indirectly imported.
    assert settings.BACKEND not in sys.modules

    # Test that flf.batch doesn't indirectly import backend.
    import flfm.batch  # noqa: F401

    assert settings.BACKEND not in sys.modules

    # Test that flf.cli doesn't indirectly import backend.
    import flfm.cli  # noqa: F401

    assert settings.BACKEND not in sys.modules

    # Test that flf.backend.base doesn't indirectly import backend.
    import flfm.backend.base  # noqa: F401

    assert settings.BACKEND not in sys.modules

    # import flfm.app.main  # This is supposed to import the backend so can't be tested.
    # assert settings.BACKEND not in sys.modules

    # Test that flf.restoration DOES directly import backend.
    import flfm.restoration  # noqa: F401

    assert settings.BACKEND in sys.modules

    # Test that other backends haven't been indirectly imported.
    for backend in BACKENDS:
        if backend != settings.BACKEND:
            assert backend not in sys.modules

    # Explicitly test that the above tests would actually fail if backend was imported.
    import flfm.backend.jax  # noqa: F401

    assert "jax" in sys.modules

    import flfm.backend.torch  # noqa: F401

    assert "torch" in sys.modules
