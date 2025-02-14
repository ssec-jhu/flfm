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


def richardson_lucy(
    image: jnp.ndarray,  # [1, n, n]
    psf: jnp.ndarray,  # [k, n, n]
    psf_fft: jnp.ndarray = None,  # [k, n, n]
    psft_fft: jnp.ndarray = None,  # [k, n, n]
    initial_guess: jnp.ndarray = None,  # [k, n, n]
    num_iter: int = 10,
    clip: bool = None,
    filter_epsilon: float = None,
    wait: bool = True,
) -> jnp.ndarray:
    """Reconstruct the image using the Richardson-Lucy deconvolution method."""

    if clip is not None or filter_epsilon is not None:
        raise NotImplementedError

    # We may want to make this something the use changes
    if initial_guess is None:
        initial_guess = jnp.ones_like(psf) * 0.5  # [k, n, n]

    if psf_fft is None:
        psf_fft = jnp.fft.rfft2(psf, axes=(-2, -1))  # [k, n, n/2+1]
    if psft_fft is None:
        psft_fft = jnp.fft.rfft2(jnp.flip(psf, axis=(-2, -1)))  # [k, n, n/2+1]

    for _ in range(num_iter):
        initial_guess = compute_step_f(initial_guess, image, psf_fft, psft_fft)

    return initial_guess.block_until_ready() if wait else initial_guess
