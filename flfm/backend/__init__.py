import importlib


def reload_backend(backend: str):
    from flfm.settings import settings

    settings.BACKEND = backend

    import flfm.io
    import flfm.restoration

    importlib.reload(flfm.io)
    importlib.reload(flfm.restoration)
