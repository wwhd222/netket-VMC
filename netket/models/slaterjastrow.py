import flax.linen as nn
import jax.numpy as jnp
from netket.utils.types import DType, Array, NNInitFunc
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.nn.masked_linear import default_kernel_init
from netket import jax as nkjax
from jax.nn.initializers import uniform
from jax.nn.initializers import normal

def custom_init(shape, dtype=jnp.float64, *args):

    return normal()(shape, dtype)

class SlaterJastrow(nn.Module):
    hilbert: SpinOrbitalFermions
    restricted: bool = True
    kernel_init: NNInitFunc = default_kernel_init
    param_dtype: DType = jnp.float64

    def setup(self):
        # Setup for Slater part
        if self.restricted:
            M = self.param(
                "M",
                self.kernel_init,
                (self.hilbert.n_orbitals, self.hilbert.n_fermions_per_spin[0]),
                self.param_dtype,
            )
            self.orbitals = [M for _ in self.hilbert.n_fermions_per_spin]
        else:
            self.orbitals = [
                self.param(
                    f"M_{i}",
                    self.kernel_init,
                    (self.hilbert.n_orbitals, nf_i),
                    self.param_dtype,
                )
                for i, nf_i in enumerate(self.hilbert.n_fermions_per_spin)
            ]

        # Setup for Jastrow part
        nv = self.hilbert.size
        self.jastrow_kernel = self.param("jastrow_kernel", custom_init, (nv * (nv - 1) // 2,), self.param_dtype)

    def log_slater(self, n):
        R = n.nonzero(size=self.hilbert.n_fermions)[0]
        log_det_sum = 0
        i_start = 0
        for i, (n_fermions_i, M_i) in enumerate(
            zip(self.hilbert.n_fermions_per_spin, self.orbitals)
        ):
            R_i = R[i_start : i_start + n_fermions_i] - i * self.hilbert.n_orbitals
            A_i = M_i[R_i]
            log_det_sum = log_det_sum + nkjax.logdet_cmplx(A_i)
            i_start = n_fermions_i

        return log_det_sum

    def log_jastrow(self, x_in: Array):
        nv = x_in.shape[-1]
        il = jnp.tril_indices(nv, k=-1)
        W = jnp.zeros((nv, nv), dtype=self.param_dtype).at[il].set(self.jastrow_kernel)
        y = jnp.einsum("...i,ij,...j", x_in, W, x_in)
        return y

    def __call__(self, n):
        if not n.shape[-1] == self.hilbert.size:
            raise ValueError(
                f"Dimension mismatch. Expected samples with {self.hilbert.size} "
                f"degrees of freedom, but got a sample of shape {n.shape}."
            )

        log_sd = self.log_slater(n)
        log_j = self.log_jastrow(n)
        return log_sd + log_j
