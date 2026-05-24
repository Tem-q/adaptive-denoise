from pathlib import Path

import torch

from .inference import classify_center, denoise_patches
from .io import from_tensor, save_image, to_tensor
from .models import NoiseClassifier, UNet


_WEIGHTS_DIR = Path(__file__).parent / "weights"

_DEFAULT_WEIGHTS = {
    "classifier": _WEIGHTS_DIR / "NoiseClassifier_best.pth",
    "real": _WEIGHTS_DIR / "unet_real.pth",
    "salt_pepper": _WEIGHTS_DIR / "unet_sp.pth",
    "speckle": _WEIGHTS_DIR / "unet_speckle.pth",
    "universal": _WEIGHTS_DIR / "unet_universal.pth",
}

VALID_NOISE_TYPES = ("real", "salt_pepper", "speckle", "universal")


def _resolve_device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class AdaptiveDenoiser:
    """Adaptive denoising system: noise-type classifier + four specialized U-Nets.

    Typical usage::

        from adaptive_denoise import AdaptiveDenoiser

        denoiser = AdaptiveDenoiser()
        clean = denoiser.denoise("noisy.png", output_path="clean.png")

    By default the noise type is detected automatically from the centre patch
    of the image. Pass ``force_type=...`` to bypass the classifier and use a
    specific model (``"real"``, ``"salt_pepper"``, ``"speckle"`` or
    ``"universal"``).
    """

    def __init__(self, device=None, patch_size=128, overlap=16):
        self.device = _resolve_device(device)
        self.patch_size = patch_size
        self.overlap = overlap

        self.classifier = NoiseClassifier().load(
            _DEFAULT_WEIGHTS["classifier"], device=self.device
        )
        self.denoisers = {
            name: UNet().load(_DEFAULT_WEIGHTS[name], device=self.device)
            for name in VALID_NOISE_TYPES
        }

    def detect_noise_type(self, image):
        """Return the predicted noise type for an image without denoising it."""
        tensor, _ = to_tensor(image)
        return classify_center(
            self.classifier, tensor, patch_size=self.patch_size, device=self.device
        )

    def denoise(self, image, force_type=None, output_path=None, return_noise_type=False):
        """Denoise an image.

        Args:
            image: input image as numpy array, PIL.Image, path or torch.Tensor.
            force_type: if given, bypass the classifier and use the named model
                (``"real"``, ``"salt_pepper"``, ``"speckle"`` or ``"universal"``).
            output_path: if given, also save the result to disk.
            return_noise_type: if True, return a ``(image, noise_type)`` tuple.

        Returns:
            The denoised image in the same format as the input (numpy/PIL/tensor).
            Path inputs return a PIL.Image.
        """
        if force_type is not None and force_type not in VALID_NOISE_TYPES:
            raise ValueError(
                f"force_type must be one of {VALID_NOISE_TYPES}, got {force_type!r}"
            )

        tensor, kind = to_tensor(image)

        if force_type is None:
            noise_type = classify_center(
                self.classifier, tensor, patch_size=self.patch_size, device=self.device
            )
        else:
            noise_type = force_type

        model = self.denoisers[noise_type]
        denoised = denoise_patches(
            model,
            tensor,
            patch_size=self.patch_size,
            overlap=self.overlap,
            device=self.device,
        )

        result = from_tensor(denoised, kind)

        if output_path is not None:
            save_image(result, output_path)

        if return_noise_type:
            return result, noise_type
        return result
