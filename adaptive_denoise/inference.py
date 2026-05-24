import torch
import torch.nn.functional as F


def _pad_to_multiple(img, multiple):
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (h, w)


def _feather_window(patch_size, overlap, device):
    w1 = torch.ones(patch_size, device=device)
    if overlap > 0:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, overlap, device=device))
        w1[:overlap] = ramp
        w1[-overlap:] = ramp.flip(0)
    return w1.unsqueeze(0) * w1.unsqueeze(1)


@torch.no_grad()
def denoise_patches(model, image, patch_size=128, overlap=16, device="cpu"):
    """Run a denoising model over an arbitrary-size image using overlapping patches
    blended with a cosine-feather window. Input must be a tensor in [0, 1] with
    shape [3, H, W] or [1, 3, H, W]; returns [3, H, W] on CPU."""
    model.eval()
    model.to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    padded, (orig_h, orig_w) = _pad_to_multiple(image, multiple=patch_size)
    _, _, ph, pw = padded.shape

    stride = patch_size - overlap
    window = _feather_window(patch_size, overlap, device)

    output = torch.zeros_like(padded)
    weight = torch.zeros((1, 1, ph, pw), device=device)

    ys = list(range(0, ph - patch_size + 1, stride))
    xs = list(range(0, pw - patch_size + 1, stride))
    if ys[-1] != ph - patch_size:
        ys.append(ph - patch_size)
    if xs[-1] != pw - patch_size:
        xs.append(pw - patch_size)

    for y in ys:
        for x in xs:
            patch = padded[:, :, y:y + patch_size, x:x + patch_size]
            pred = model(patch).clamp(0, 1)
            output[:, :, y:y + patch_size, x:x + patch_size] += pred * window
            weight[:, :, y:y + patch_size, x:x + patch_size] += window

    output = output / weight.clamp(min=1e-8)
    output = output[:, :, :orig_h, :orig_w]
    return output.squeeze(0).cpu()


@torch.no_grad()
def classify_center(classifier, image, patch_size=128, device="cpu"):
    """Classify noise type on the centre patch of an image.
    Returns the predicted noise-type name (string)."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    _, _, h, w = image.shape
    cy = max(0, h // 2 - patch_size // 2)
    cx = max(0, w // 2 - patch_size // 2)
    center = image[:, :, cy:cy + patch_size, cx:cx + patch_size]

    _, names = classifier.predict_noise_type(center, device=device)
    return names[0]
