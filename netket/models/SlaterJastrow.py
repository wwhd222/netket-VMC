import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import normal
from functools import partial

from netket.utils.types import DType, Array, NNInitFunc
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.nn.masked_linear import default_kernel_init
from netket import jax as nkjax

class SlaterJastrow(nn.Module):
    hilbert: SpinOrbitalFermions
    kernel_init: NNInitFunc = normal()
    param_dtype: DType = jnp.complex128
    restricted: bool = True  # Assuming a restricted setting for simplicity

    def setup(self):
        nv = self.hilbert.size
        self.kernel = self.param("kernel", self.kernel_init, (nv, nv), self.param_dtype)
        self.kernel = self.kernel + self.kernel.T

        # Setting up the orbital parameters (assuming restricted setting)
        self.M = self.param(
            "M",
            self.kernel_init,
            (self.hilbert.n_orbitals, self.hilbert.n_fermions_per_spin[0]),
            self.param_dtype,
        )
        self.orbitals = [self.M for _ in self.hilbert.n_fermions_per_spin]

    def log_jastrow(self, x_in: Array):
        kernel, x_in = nkjax.promote_dtype(self.kernel, x_in, dtype=None)
        y = jnp.einsum("...i,ij,...j", x_in, kernel, x_in)
        return y

    def log_slater_determinant(self, n):
        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            log_det_sum = 0
            i_start = 0
            for i, (n_fermions_i, M_i) in enumerate(
                zip(self.hilbert.n_fermions_per_spin, self.orbitals)
            ):
                R_i = R[i_start : i_start + n_fermions_i] - i * self.hilbert.n_orbitals
                A_i = M_i[R_i]
                log_det_sum = log_det_sum + nkjax.logdet_cmplx(A_i)
                i_start += n_fermions_i

            return log_det_sum

        return log_sd(n)

    def __call__(self, n):
        if not n.shape[-1] == self.hilbert.size:
            raise ValueError(
                f"Dimension mismatch. Expected samples with {self.hilbert.size} "
                f"degrees of freedom, but got a sample of shape {n.shape}."
            )

        # Compute Jastrow term
        y = self.log_jastrow(n)

        # Compute log-slater determinant
        log_sd = self.log_slater_determinant(n)

        return log_sd + y
