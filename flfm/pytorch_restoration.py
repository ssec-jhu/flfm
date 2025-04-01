"""Deconvolution for FLFM observation reconstruction."""

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
