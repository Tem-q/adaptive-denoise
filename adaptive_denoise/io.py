from pathlib import Path

import numpy as np
import torch
from PIL import Image


ImageInput = "np.ndarray | Image.Image | str | Path | torch.Tensor"


def to_tensor(image):
    """Convert input (numpy, PIL, path, tensor) to a float tensor of shape [3, H, W] in [0, 1].
    Also returns the original input kind so the result can be converted back to the same type."""
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1), "pil"

    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1), "pil"

    if isinstance(image, np.ndarray):
        kind = "numpy_uint8" if image.dtype == np.uint8 else "numpy_float"
        arr = image.astype(np.float32)
        if image.dtype == np.uint8:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return torch.from_numpy(arr).permute(2, 0, 1), kind

    if isinstance(image, torch.Tensor):
        t = image.float()
        if t.dim() == 4:
            t = t.squeeze(0)
        return t, "tensor"

    raise TypeError(f"Unsupported image type: {type(image)}")


def from_tensor(tensor, kind):
    """Convert a [3, H, W] tensor in [0, 1] back to the requested output kind."""
    tensor = tensor.clamp(0, 1).cpu()
    arr = tensor.permute(1, 2, 0).numpy()

    if kind == "tensor":
        return tensor
    if kind == "numpy_float":
        return arr.astype(np.float32)
    if kind == "numpy_uint8":
        return (arr * 255.0).round().astype(np.uint8)
    if kind == "pil":
        return Image.fromarray((arr * 255.0).round().astype(np.uint8))

    raise ValueError(f"Unknown kind: {kind}")


def save_image(image, path):
    """Save a PIL Image, numpy array or tensor to disk."""
    if isinstance(image, Image.Image):
        image.save(path)
        return
    if isinstance(image, torch.Tensor):
        image = from_tensor(image, "pil")
        image.save(path)
        return
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image.clip(0, 1) * 255.0).round().astype(np.uint8)
        Image.fromarray(image).save(path)
        return
    raise TypeError(f"Unsupported image type: {type(image)}")
