import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import normal
from functools import partial
import numpy as np

from netket.utils.types import DType, Array, NNInitFunc
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.nn.masked_linear import default_kernel_init
from netket import jax as nkjax

class slater_jastrow(nn.Module):
    hilbert: SpinOrbitalFermions
    kernel_init: NNInitFunc = normal()
    param_dtype: DType = jnp.complex128
    restricted: bool = True  # Assuming a restricted setting for simplicity

    def __post_init__(self):
        if not isinstance(self.hilbert, SpinOrbitalFermions):
            raise TypeError(
                "Slater2nd only supports 2nd quantised fermionic hilbert spaces."
            )
        if self.hilbert.n_fermions is None:
            raise TypeError(
                "Slater2nd only supports hilbert spaces with a "
                "fixed number of fermions."
            )
        if self.restricted:
            if not all(
                np.equal(
                    self.hilbert.n_fermions_per_spin,
                    self.hilbert.n_fermions_per_spin[0],
                )
            ):
                raise ValueError(
                    "Restricted Hartree Fock only makes sense for spaces with "
                    "same number of fermions on every subspace."
                )
        super().__post_init__()

    def setup(self):
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

    def log_jastrow(self, x_in: Array):
        dtype = jnp.result_type(self.kernel, x_in)
        kernel = self.kernel.astype(dtype)
        x_in = x_in.astype(dtype)
        y = jnp.einsum("...i,ij,...j", x_in, kernel, x_in)
        return y

    def __call__(self, n):
        """
        Assumes inputs are strings of 0,1 that specify which orbitals are occupied.
        Spin sectors are assumed to follow the SpinOrbitalFermion's factorisation,
        meaning that the first `n_orbitals` entries correspond to sector -1, the
        second `n_orbitals` correspond to 0 ... etc.
        """
        if not n.shape[-1] == self.hilbert.size:
            raise ValueError(
                f"Dimension mismatch. Expected samples with {self.hilbert.size} "
                f"degrees of freedom, but got a sample of shape {n.shape} ({n.shape[-1]} dof)."
            )

        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            # Find the positions of the occupied sites
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            log_det_sum = 0
            i_start = 0
            for i, (n_fermions_i, M_i) in enumerate(
                zip(self.hilbert.n_fermions_per_spin, self.orbitals)
            ):
                # convert global orbital positions to spin-sector-local positions
                R_i = R[i_start : i_start + n_fermions_i] - i * self.hilbert.n_orbitals
                # extract the corresponding Nf x Nf submatrix
                A_i = M_i[R_i]

                log_det_sum = log_det_sum + nkjax.logdet_cmplx(A_i)
                i_start = n_fermions_i

            return log_det_sum

        return log_sd(n)
        
        # Compute Jastrow term
        y = self.log_jastrow(n)

        # Compute log-slater determinant
        log_sd = self.log_sd(n)

        return log_sd + y

