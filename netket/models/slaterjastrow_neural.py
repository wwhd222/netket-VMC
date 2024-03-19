import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.utils.types import DType, Array, NNInitFunc
from functools import partial

def complex_normal(key, shape, dtype=jnp.complex128, scale=0.01):
    """Complex normal initializer with scale."""
    real_key, imag_key = jax.random.split(key)
    real_part = scale * jax.random.normal(real_key, shape)
    imag_part = scale * jax.random.normal(imag_key, shape)
    return (real_part + 1j * imag_part).astype(dtype)

class ComplexDense(nn.Module):
    features: int
    kernel_init: NNInitFunc
    dtype: DType = jnp.complex128

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features), self.dtype)
        return jnp.dot(inputs, kernel)

class SlaterJastrowComplex(nn.Module):
    hilbert: SpinOrbitalFermions
    hidden_units: int
    dtype: DType = jnp.complex128

    def setup(self):
        self.M = self.param(
            "M",
            complex_normal,  # Use the complex_normal with the scale parameter
            (self.hilbert.n_orbitals, self.hilbert.n_fermions_per_spin[0]),
            self.dtype
        )
        self.orbitals = [self.M for _ in self.hilbert.n_fermions_per_spin]
        self.jastrow_network = ComplexDense(self.hidden_units, kernel_init=complex_normal, dtype=self.dtype)

    def log_slater_determinant(self, n: Array):
        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            log_det_sum = 0
            i_start = 0
            for i, n_fermions_i in enumerate(self.hilbert.n_fermions_per_spin):
                R_i = R[i_start : i_start + n_fermions_i] - i * self.hilbert.n_orbitals
                A_i = self.M[R_i]
                log_det_sum += nkjax.logdet_cmplx(A_i)
                i_start += n_fermions_i
            return log_det_sum

        return log_sd(n)

    def log_jastrow(self, n: Array):
        return jnp.sum(self.jastrow_network(n))

    def __call__(self, n: Array):
        if not n.shape[-1] == self.hilbert.size:
            raise ValueError(
                f"Dimension mismatch. Expected samples with {self.hilbert.size} "
                f"degrees of freedom, but got a sample of shape {n.shape}."
            )

        log_sd = self.log_slater_determinant(n)
        log_j = self.log_jastrow(n)

        return log_sd + log_j
