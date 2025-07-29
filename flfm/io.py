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
    case "numpy" | "cupy" | "cupynumeric":
        from flfm.backend.numpy import NumpyIO

        IO = NumpyIO()
        assert IO is NumpyIO()
    case _:
        raise NotImplementedError(f"Unsupported backend: '{settings.BACKEND}'")


open = IO.open
save = IO.save
