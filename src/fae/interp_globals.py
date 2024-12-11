from contextlib import contextmanager
from types import SimpleNamespace

# interp_handlers = threading.local()
interp_handlers = SimpleNamespace()


@contextmanager
def set_contextvar(key, value):
    old_val = getattr(interp_handlers, key, None)
    setattr(interp_handlers, key, value)
    try:
        yield
    finally:
        setattr(interp_handlers, key, old_val)
