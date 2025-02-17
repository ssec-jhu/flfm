"""Deconvolution for FLFM observation reconstruction."""

import functools

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
    img_err = image / denom  # [1, n, n]
    return data * jnp.fft.fftshift(jnp.fft.irfft2(jnp.fft.rfft2(img_err) * PSFt_fft), axes=(-2, -1))  # [k, n, n]


@functools.partial(jax.jit, static_argnames=("n_iters",))
def _compute_step_java(
    data: jnp.ndarray,  # [k, n, n] # noqa: E741
    image: jnp.ndarray,  # [1, n, n] # noqa: E741
    psf: jnp.ndarray,  # [k, n, n]
    n_iters: int,
) -> jnp.ndarray:
    psft_fft = jnp.fft.rfft2(jnp.flip(psf, axis=(-2, -1)))  # [k, n, n/2+1]
    psf_fft = jnp.fft.rfft2(psf)  # [k, n, n/2+1]

    for _ in range(n_iters):
        data = compute_step_f(data, image, psf_fft, psft_fft)

    return data


def reconstruct(
    image: jnp.ndarray,  # [1, n, n]
    psf: jnp.ndarray,  # [k, n, n]
    num_iter: int = 10,
    **kwargs,
) -> jnp.ndarray:
    """Reconstruct the image using the Richardson-Lucy deconvolution method."""

    if "clip" in kwargs or "filter_epsilon" in kwargs:
        raise NotImplementedError

    # We may want to make this something the use changes
    data = jnp.ones_like(psf) * 0.5  # [k, n, n]

    psf_fft = jnp.fft.rfft2(psf, axes=(-2, -1))  # [k, n, n/2+1]
    psft_fft = jnp.fft.rfft2(jnp.flip(psf, axis=(-2, -1)))  # [k, n, n/2+1]

    for _ in range(num_iter):
        data = compute_step_f(data, image, psf_fft, psft_fft).block_until_ready()

    return data


def export_to_tf(out_path: str):
    import tensorflow as tf
    from jax.experimental import jax2tf

    exported_f = tf.Module()
    exported_f.f = tf.function(
        jax2tf.convert(
            functools.partial(_compute_step_java, n_iters=10),
            with_gradient=False,
            native_serialization_platforms=("cuda",),
        ),
        autograph=False,
        input_signature=[
            tf.TensorSpec(shape=[41, 2048, 2048], dtype=tf.float32, name="data"),  # [k, n, n]
            tf.TensorSpec(shape=[1, 2048, 2048], dtype=tf.float32, name="image"),  # [1, n, n]
            tf.TensorSpec(shape=[41, 2048, 2048], dtype=tf.float32, name="psf"),  # [k, n, n]
        ],
    )

    exported_f.f(
        tf.ones((41, 2048, 2048), dtype=tf.float32, name="data"),
        tf.ones((1, 2048, 2048), dtype=tf.float32, name="image"),
        tf.ones((41, 2048, 2048), dtype=tf.float32, name="psf"),
    )

    tf.saved_model.save(exported_f, out_path)
