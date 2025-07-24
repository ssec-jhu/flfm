"""Image/data io operations."""

from flfm.settings import settings

match settings.BACKEND:
    case "jax":
        from flfm.backend.jax import JaxIO

        IO = JaxIO()
        assert IO is JaxIO()
    case "torch":
        from flfm.backend.torch import TorchIO

        IO = TorchIO()
        assert IO is TorchIO()
    case "numpy":
        from flfm.backend.numpy import NumpyIO

        IO = NumpyIO()
        assert IO is NumpyIO()
    case _:
        raise NotImplementedError


open = IO.open
save = IO.save
