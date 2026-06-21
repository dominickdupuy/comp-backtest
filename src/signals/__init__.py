"""Signal package. Importing it registers all built-in signals."""
from . import bab, lottery, momentum, overnight, pead, reversal  # noqa: F401
from .base import (  # noqa: F401
    DataBundle,
    Signal,
    available_signals,
    build_signal,
    register,
)
