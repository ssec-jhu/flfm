"""Deconvolution for FLFM observation reconstruction."""

from pathlib import Path

import torch


@torch.jit.script
def compute_step_f(
    data: torch.Tensor,  # [k, n, n]
    image: torch.Tensor,  # [1, n, n]
    PSF_fft: torch.Tensor,  # [k, n, n/2+1]
    PSFt_fft: torch.Tensor,  # [k, n, n/2+1]
) -> torch.Tensor:
    """Single step of the multiplicative Richardson-Lucy deconvolution algorithm."""
    denom = torch.fft.irfft2(PSF_fft * torch.fft.rfft2(data), dim=(-2, -1)).sum(dim=0, keepdim=True)  # [1, n, n]
    img_err = image / denom
    return data * torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(img_err) * PSFt_fft), dim=(-2, -1))  # [k, n, n]


def richardson_lucy(
    image: torch.Tensor,  # [1, n, n]
    psf: torch.Tensor,  # [k, n, n]
    num_iter: int = 10,
    **kwargs,
) -> torch.Tensor:
    """Reconstruct the image using the Richardson-Lucy deconvolution method."""

    if torch.cuda.is_available():
        # Move data to GPU.
        image = image.to("cuda")
        psf = psf.to("cuda")

    if "clip" in kwargs or "filter_epsilon" in kwargs:
        raise NotImplementedError

    # We may want to make this something the use changes
    data = torch.ones_like(psf) * 0.5  # [k, n, n]

    psf_fft = torch.fft.rfft2(psf, dim=(-2, -1))  # [k, n, n/2+1]
    psft_fft = torch.fft.rfft2(torch.flip(psf, dims=(-2, -1)))  # [k, n, n/2+1]

    for _ in range(num_iter):
        data = compute_step_f(data, image, psf_fft, psft_fft)

    return data.cpu()


def unroll_and_save(
    num_steps: int,
    out_path: str | Path,
    img_size: tuple[int, int, int] = (),
    psf_size: tuple[int, int, int] = (),
) -> None:
    """Unroll the Richardson-Lucy algorithm for a given number of steps amd save it."""

    def rl(img: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        psf_fft = torch.fft.rfft2(psf)  # [k, n, n/2+1]
        psft_fft = torch.fft.rfft2(torch.flip(psf))  # [k, n, n/2+1]
        data = torch.ones_like(psf) * 0.5  # [k, n, n]

        for _ in range(num_steps):
            data = compute_step_f(data, img, psf_fft, psft_fft)

        return data

    jitted_fn = torch.jit.script(
        rl,
        example_inputs=(
            torch.zeros(*img_size, dtype=torch.float32),
            torch.zeros(*psf_size, dtype=torch.float32),
        ),
    )

    torch.jit.save(jitted_fn, out_path)
