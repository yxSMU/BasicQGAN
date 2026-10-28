# models/size28_past_tc_analyze_PiorApplyPast_FullUnitary_noanc.py
import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorcircuit as tc
import numpy as np

tc.set_backend("tensorflow")


class ClassicalCritic(Model):
    """
    critic：784 -> 512 -> 256 -> 1
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = layers.Dense(512)
        self.fc2 = layers.Dense(256)
        self.fc3 = layers.Dense(1)

    def call(self, x, training=False):
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = tf.nn.leaky_relu(self.fc1(x), 0.2)
        x = tf.nn.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)


class QuantumGeneratorFullUnitaryNoAncilla(Model):
    """
    - FullUnitary: RY-RZ-RY + CNOT chain
    """

    def __init__(
        self, n_qubits: int, n_layers: int, output_dim: int, use_born_prob: bool = True
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.output_dim = int(output_dim)
        self.use_born_prob = bool(use_born_prob)

        full_dim = 2**self.n_qubits
        if full_dim < self.output_dim:
            raise ValueError(
                f"2**n_qubits must be >= output_dim. Got {full_dim} < {self.output_dim}"
            )

        self.tc_layer = tc.KerasLayer(
            self._circuit_state,
            weights_shape=(self.n_layers, self.n_qubits, 1),
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.0, maxval=2.0 * np.pi
            ),
        )

    def _circuit_state(self, latent_vec_1d, weights):
        c = tc.Circuit(self.n_qubits)

        for q in range(self.n_qubits):
            c.ry(q, theta=latent_vec_1d[q])

        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                c.ry(q, theta=weights[l, q, 0])
            for q in range(self.n_qubits - 1):
                c.cnot(q, q + 1)

        return c.state()

    def call(self, z, training=False):
        z = tf.cast(z, tf.float32)
        state = self.tc_layer(z, training=training)  # [B, 2**n_qubits] complex

        amplitudes = tf.abs(state)
        maxv = tf.reduce_max(amplitudes, axis=1, keepdims=True) + 1e-12
        post_processed = 2.0 * amplitudes / maxv - 1.0

        return post_processed[:, : self.output_dim]


class PQWGAN_CC(Model):
    """ """

    def __init__(
        self,
        image_size: int = 28,
        channels: int = 1,
        n_data_qubits: int = 10,
        n_layers: int = 40,
        use_born_prob: bool = True,
    ):
        super().__init__()
        if channels != 1:
            raise ValueError("Only single-channel grayscale images are supported.")

        self.image_size = int(image_size)
        self.output_dim = self.image_size * self.image_size  # 784

        self.n_data_qubits = int(n_data_qubits)
        self.n_qubits = self.n_data_qubits  # no ancilla

        self.critic = ClassicalCritic(self.output_dim)
        self.generator = QuantumGeneratorFullUnitaryNoAncilla(
            n_qubits=self.n_qubits,
            n_layers=n_layers,
            output_dim=self.output_dim,
            use_born_prob=use_born_prob,
        )
