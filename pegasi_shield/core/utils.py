import logging

log = logging.getLogger(__name__)

# Internal Utility Functions
# These functions are for internal use and not part of the public API.

# Detect the PyTorch device
try:
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    device_int = 0 if device.type == "cuda" else -1
except ImportError:
    # Fallback for when accelerate is not installed
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_int = 0 if device.type == "cuda" else -1
    except ImportError:
        # Final fallback when neither accelerate nor torch is available
        device = "cpu"
        device_int = -1
