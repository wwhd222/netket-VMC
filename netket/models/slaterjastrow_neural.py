
import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import complex_normal  # Assuming a complex initializer exists

class ComplexDense(nn.Module):
    features: int
    kernel_init: NNInitFunc = complex_normal()
    dtype: DType = jnp.complex128

    def setup(self):
        # Initialize the complex-valued weights
        self.kernel = self.param('kernel',
                                 self.kernel_init,
                                 (self.features,),
                                 self.dtype)

    def __call__(self, inputs):
        # Perform the matrix multiplication with complex numbers
        return jnp.dot(inputs, self.kernel)

class SlaterJastrowComplex(nn.Module):
    hilbert: SpinOrbitalFermions
    hidden_units: int
    kernel_init: NNInitFunc = complex_normal()
    param_dtype: DType = jnp.complex128  # Use complex numbers for parameters

    def setup(self):
        # Initialize the neural network for the Jastrow factor with complex parameters
        self.dense_jastrow = ComplexDense(
            self.hidden_units,
            dtype=self.param_dtype
        )

        # Setting up the Slater determinant part
        self.M = self.param(
            "M",
            self.kernel_init,
            (self.hilbert.n_orbitals, self.hilbert.n_fermions_per_spin[0]),
            self.param_dtype
        )
        self.orbitals = [self.M for _ in self.hilbert.n_fermions_per_spin]

    def log_jastrow(self, n: Array):
        # Compute the Jastrow term using the neural network
        J = self.dense_jastrow(n)
        # Apply an activation function appropriate for complex numbers
        J = jnp.tanh(jnp.abs(J))  # Example: apply tanh to the magnitude
        return J.sum()

    def log_slater_determinant(self, n):
        # Compute the Slater determinant part
        R = n.nonzero(size=self.hilbert.n_fermions)[0]
        log_det_sum = 0
        i_start = 0
        for i, (n_fermions_i, M_i) in enumerate(
                zip(self.hilbert.n_fermions_per_spin, self.orbitals)):
            R_i = R[i_start : i_start + n_fermions_i] - i * self.hilbert.n_orbitals
            A_i = M_i[R_i]
            log_det_sum += nkjax.logdet_cmplx(A_i)
            i_start += n_fermions_i

        return log_det_sum

    def __call__(self, n):
        # Check if the input size matches the Hilbert space size
        if not n.shape[-1] == self.hilbert.size:
            raise ValueError(
                f"Dimension mismatch. Expected samples with {self.hilbert.size} "
                f"degrees of freedom, but got a sample of shape {n.shape}."
            )

        # Compute the Jastrow and Slater determinant terms
        jastrow_term = self.log_jastrow(n)
        log_sd = self.log_slater_determinant(n)

        return log_sd + jastrow_term

