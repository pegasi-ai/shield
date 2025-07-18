# pegasi_shield/__init__.py
from .core.engine import _CoreShieldEngine
from .shield import Shield, ShieldError, Detector

# (Optionally) expose existing internal detectors:
# from .core.input_detectors import ...
# from .core.output_detectors import ...

__all__ = ["Shield", "ShieldError", "Detector", "_CoreShieldEngine"]
