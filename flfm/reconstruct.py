"""Deconvolution for FLFM observation reconstruction."""

import jax
import jax.numpy as jnp


@jax.jit
def compute_step_f(
    O: jnp.ndarray,  # [k, n, n] # noqa: E741
    I: jnp.ndarray,  # [1, n, n] # noqa: E741
    PSF_fft: jnp.ndarray,  # [k, n, n/2+1]
    PSFt_fft: jnp.ndarray,  # [k, n, n/2+1]
) -> jnp.ndarray:
    """Single step of the multiplicative Richardson-Lucy deconvolution algorithm."""
    denom = jnp.fft.irfft2(PSF_fft * jnp.fft.rfft2(O)).sum(axis=0, keepdims=True)  # [1, n, n]
    img_err = I / denom
    return O * jnp.fft.fftshift(jnp.fft.irfft2(jnp.fft.rfft2(img_err) * PSFt_fft), axes=(-2, -1))  # [k, n, n]


def reconstruct(
    I: jnp.ndarray,  # [1, n, n] # noqa: E741
    PSF: jnp.ndarray,  # [k, n, n] # noqa: E741
    num_iter: int = 10,
) -> jnp.ndarray:
    """Reconstruct the image using the deconvolution method."""
    # We may want to make this something the use changes
    O = jnp.ones_like(PSF) * 0.5  # noqa: E741 [k, n, n]

    PSF_fft = jnp.fft.rfft2(PSF, axes=(-2, -1))  # [k, n, n/2+1]
    PSFt_fft = jnp.fft.rfft2(jnp.flip(PSF, axis=(-2, -1)))  # [k, n, n/2+1]

    for _ in range(num_iter):
        O = compute_step_f(O, I, PSF_fft, PSFt_fft).block_until_ready()  # noqa: E741

    return O
