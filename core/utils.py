import logging

from accelerate import Accelerator

log = logging.getLogger(__name__)

# Internal Utility Functions
# These functions are for internal use and not part of the public API.

# Detect the PyTorch device
accelerator = Accelerator()
device = accelerator.device
device_int = 0 if device.type == "cuda" else -1
