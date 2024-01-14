import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import normal
from functools import partial
import numpy as np

from netket.utils.types import DType, Array, NNInitFunc
from netket.experimental.hilbert import SpinOrbitalFermions
from netket import jax as nkjax

class slater_jastrow(nn.Module):
    hilbert: SpinOrbitalFermions
    jastrow_kernel_init: NNInitFunc = normal()
    slater_kernel_init: NNInitFunc = normal()
    param_dtype: DType = jnp.complex128
    restricted: bool = True  # Assuming a restricted setting for simplicity

    def setup(self):
        # Slater part setup
        if self.restricted:
            self.M = self.param(
                "M",
                self.slater_kernel_init,
                (self.hilbert.n_orbitals, self.hilbert.n_fermions_per_spin[0]),
                self.param_dtype,
            )
            self.orbitals = [self.M for _ in self.hilbert.n_fermions_per_spin]
        else:
            self.orbitals = [
                self.param(
                    f"M_{i}",
                    self.slater_kernel_init,
                    (self.hilbert.n_orbitals, nf_i),
                    self.param_dtype,
                )
                for i, nf_i in enumerate(self.hilbert.n_fermions_per_spin)
            ]

        # Jastrow part setup
        nv = self.hilbert.size
        self.jastrow_kernel = self.param(
            "jastrow_kernel", 
            self.jastrow_kernel_init, 
            (nv * (nv - 1) // 2,), 
            self.param_dtype
        )

    def log_jastrow(self, x_in: Array):
        nv = x_in.shape[-1]
        il = jnp.tril_indices(nv, k=-1)

        # Retrieve and reshape Jastrow kernel
        jastrow_kernel = self.jastrow_kernel
        W = jnp.zeros((nv, nv), dtype=self.param_dtype).at[il].set(jastrow_kernel)

        # Ensure W and x_in have the same data type
        W = W.astype(x_in.dtype)
        x_in = x_in.astype(W.dtype)

        y = jnp.einsum("...i,ij,...j", x_in, W, x_in)
        return y


    def log_sd(self, n):
        # Compute Slater determinant
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

    def __call__(self, n):
        if not n.shape[-1] == self.hilbert.size:
            raise ValueError(
                f"Dimension mismatch. Expected samples with {self.hilbert.size} "
                f"degrees of freedom, but got a sample of shape {n.shape}."
            )

        log_sd_val = self.log_sd(n)  # Compute log-Slater determinant
        y = self.log_jastrow(n)      # Compute Jastrow term

        return log_sd_val + y
