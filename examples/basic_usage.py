"""Minimal usage example: load a noisy image, denoise it, save the result."""

from adaptive_denoise import AdaptiveDenoiser


def main():
    denoiser = AdaptiveDenoiser()

    clean, noise_type = denoiser.denoise(
        "noisy.png",
        output_path="clean.png",
        return_noise_type=True,
    )
    print(f"Detected noise type: {noise_type}")
    print("Result saved to clean.png")


if __name__ == "__main__":
    main()
