"""Deconvolution for FLFM observation reconstruction."""

import jax
import jax.numpy as jnp


@jax.jit
def compute_step_f(
    data: jnp.ndarray,  # [k, n, n]
    image: jnp.ndarray,  # [1, n, n]
    PSF_fft: jnp.ndarray,  # [k, n, n/2+1]
    PSFt_fft: jnp.ndarray,  # [k, n, n/2+1]
) -> jnp.ndarray:
    """Single step of the multiplicative Richardson-Lucy deconvolution algorithm."""
    denom = jnp.fft.irfft2(PSF_fft * jnp.fft.rfft2(data)).sum(axis=0, keepdims=True)  # [1, n, n]
    img_err = image / denom
    return data * jnp.fft.fftshift(jnp.fft.irfft2(jnp.fft.rfft2(img_err) * PSFt_fft), axes=(-2, -1))  # [k, n, n]


def reconstruct(
    image: jnp.ndarray,  # [1, n, n]
    psf: jnp.ndarray,  # [k, n, n]
    num_iter: int = 10,
) -> jnp.ndarray:
    """Reconstruct the image using the deconvolution method."""
    # We may want to make this something the use changes
    data = jnp.ones_like(psf) * 0.5  # [k, n, n]

    psf_fft = jnp.fft.rfft2(psf, axes=(-2, -1))  # [k, n, n/2+1]
    psft_fft = jnp.fft.rfft2(jnp.flip(psf, axis=(-2, -1)))  # [k, n, n/2+1]

    for _ in range(num_iter):
        data = compute_step_f(data, image, psf_fft, psft_fft).block_until_ready()

    return data
