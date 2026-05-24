# adaptive-denoise

Adaptive image denoising system. A lightweight noise-type classifier routes
each image to one of several specialized U-Net denoisers, each trained on a
single type of noise. Specialization yields better PSNR/SSIM than a single
universal model trained on a mix of noise types.

Supported noise types:

- `real` — real sensor noise (trained on the SIDD dataset)
- `salt_pepper` — synthetic impulse noise
- `speckle` — synthetic multiplicative noise

A `universal` model trained on a mix of all three is also included as a
fallback when the noise type is unknown or out of distribution.

## Installation

```bash
pip install git+https://github.com/Tem-q/adaptive-denoise.git
```

Pretrained weights (~120 MB) are bundled with the package, no extra download
step needed.

## Quick start

```python
from adaptive_denoise import AdaptiveDenoiser

denoiser = AdaptiveDenoiser()
clean = denoiser.denoise("noisy.png", output_path="clean.png")
```

The output type mirrors the input:

```python
import numpy as np
from PIL import Image
from adaptive_denoise import AdaptiveDenoiser

denoiser = AdaptiveDenoiser()

denoiser.denoise(np.zeros((512, 512, 3), dtype=np.uint8))     # -> np.ndarray (uint8)
denoiser.denoise(Image.open("a.png"))                         # -> PIL.Image
denoiser.denoise("a.png")                                     # -> PIL.Image
```

### Bypass the classifier

If you already know the noise type, pass `force_type`:

```python
clean = denoiser.denoise("noisy.png", force_type="speckle")
```

Valid values: `"real"`, `"salt_pepper"`, `"speckle"`, `"universal"`.

### Inspect the classifier

```python
denoiser.detect_noise_type("noisy.png")           # -> 'real'

clean, noise_type = denoiser.denoise(
    "noisy.png", return_noise_type=True
)
```

## API

### `AdaptiveDenoiser(device=None, patch_size=128, overlap=16)`

- `device`: `"cpu"`, `"cuda"`, or `None` to auto-select.
- `patch_size`: size of the patches the U-Nets were trained on. Must be 128 for
  the bundled weights.
- `overlap`: patch overlap in pixels for seamless stitching of full-resolution
  images.

### `denoise(image, force_type=None, output_path=None, return_noise_type=False)`

Inputs accepted: `numpy.ndarray` (uint8 or float), `PIL.Image`, file path
(`str` or `pathlib.Path`), or `torch.Tensor` shaped `[3, H, W]` or
`[1, 3, H, W]` with values in `[0, 1]`.

The output matches the input type. If `output_path` is given the result is
additionally saved to disk.

### `detect_noise_type(image)`

Returns the predicted noise type as a string, without running denoising.

## How it works

1. The classifier looks at the centre 128×128 patch of the image and predicts
   one of the three noise types. This is cheap (~2.4 MB model) and the noise
   type is assumed homogeneous across the image.
2. The corresponding specialized U-Net is applied to the whole image using
   overlapping patches blended with a cosine-feather window, so there are no
   visible seams on arbitrary-resolution inputs.

## License

MIT
