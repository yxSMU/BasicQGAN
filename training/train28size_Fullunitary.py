# train_pqwgan_tf_tc_prob28_uniform_noanc.py
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import argparse
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import tensorcircuit as tc

tc.set_backend("tensorflow")

from utils.dataset_gpu import load_mnist_28_intensity, select_from_dataset
from models.size28_past_tc_analyze_PiorApplyPast_FullUnitary import PQWGAN_CC


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[Warn] No GPU found. Running on CPU.")
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[OK] Using GPU(s): {gpus}")


def denorm_to_torch_28(x_tf_flat: tf.Tensor) -> torch.Tensor:
    """
    [-1,1] -> [0,1]
    """
    x = tf.reshape(x_tf_flat, (-1, 1, 28, 28))
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(x, 0.0, 1.0)
    return torch.from_numpy(x.numpy())


def compute_gradient_penalty_tf(critic, real_x, fake_x):
    bs = tf.shape(real_x)[0]
    alpha = tf.random.uniform([bs, 1], 0.0, 1.0, dtype=tf.float32)
    inter = alpha * real_x + (1.0 - alpha) * fake_x

    with tf.GradientTape() as tape:
        tape.watch(inter)
        v = critic(inter, training=True)

    grads = tape.gradient(v, inter)
    grads = tf.reshape(grads, [bs, -1])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    gp = tf.reduce_mean(tf.square(slopes - 1.0))
    return gp


def train(
    classes_str: str,
    n_layers: int,
    n_data_qubits: int,
    batch_size: int,
    out_dir: str,
    per_class_size: int = 1000,
    n_epochs: int = 50,
):
    setup_gpu()
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(list(set(int(d) for d in classes_str.split(","))))

    image_size = 28
    output_dim = image_size * image_size  # 784

    lr_D = 0.0002
    lr_G = 0.01
    b1, b2 = 0.0, 0.9
    lambda_gp = 10.0
    n_critic = 5
    sample_interval = 10

    latent_dim = n_data_qubits

    if latent_dim != 10:
        raise ValueError(f"Expect n_data_qubits/latent_dim = 10, but got {latent_dim}.")

    a, b, c, d, e, f, g, h, i, j = (
        0.44475609064102173,
        0.3188556432723999,
        0.4070402681827545,
        0.4302237033843994,
        0.384520024061203,
        0.37214961647987366,
        0.41762977838516235,
        0.446796178817749,
        0.3816107213497162,
        0.3397783637046814,
    )
    scales = (
        tf.constant([a, b, c, d, e, f, g, h, i, j], dtype=tf.float32) * np.pi
    )  # [10]

    def sample_z(bs: tf.Tensor) -> tf.Tensor:
        z_rand = tf.random.uniform([bs, 10], 0.0, 1.0, dtype=tf.float32)
        z = z_rand * tf.reshape(scales, [1, 10])
        return z  # [bs, 10]

        # Dataset：[-1,1] & shape=[784]

    dataset = select_from_dataset(load_mnist_28_intensity(), per_class_size, classes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    gan = PQWGAN_CC(
        image_size=image_size,
        channels=1,
        n_data_qubits=n_data_qubits,  # 10
        n_layers=n_layers,
        use_born_prob=True,
    )
    critic = gan.critic
    generator = gan.generator

    _ = critic(tf.zeros((1, output_dim), dtype=tf.float32))
    _ = generator(tf.zeros((1, latent_dim), dtype=tf.float32), training=False)

    opt_C = tf.keras.optimizers.Adam(learning_rate=lr_D, beta_1=b1, beta_2=b2)
    opt_G = tf.keras.optimizers.Adam(learning_rate=lr_G, beta_1=b1, beta_2=b2)

    @tf.function
    def critic_step(real_x):
        bs = tf.shape(real_x)[0]
        with tf.GradientTape() as tape:
            z = sample_z(bs)
            fake_x = generator(z, training=True)

            real_v = critic(real_x, training=True)
            fake_v = critic(fake_x, training=True)

            gp = compute_gradient_penalty_tf(critic, real_x, fake_x)
            d_loss = -tf.reduce_mean(real_v) + tf.reduce_mean(fake_v) + lambda_gp * gp

        grads = tape.gradient(d_loss, critic.trainable_variables)
        opt_C.apply_gradients(zip(grads, critic.trainable_variables))

        wdist = tf.reduce_mean(real_v) - tf.reduce_mean(fake_v)
        return d_loss, wdist

    @tf.function
    def generator_step(bs):
        with tf.GradientTape() as tape:
            z = sample_z(bs)
            fake_x = generator(z, training=True)
            fake_v = critic(fake_x, training=True)
            g_loss = -tf.reduce_mean(fake_v)

        grads = tape.gradient(g_loss, generator.trainable_variables)
        opt_G.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss

    fixed_z = sample_z(tf.constant(batch_size, dtype=tf.int32))

    wasserstein_distance_history = []
    saved_initial = False
    batches_done = 0

    for epoch in range(n_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for i_batch, (real_x_pt, _) in enumerate(pbar):
            real_x = tf.convert_to_tensor(real_x_pt.numpy(), dtype=tf.float32)
            bs = tf.shape(real_x)[0]

            if not saved_initial:
                fixed_images = generator(fixed_z[:bs], training=False)
                save_image(
                    denorm_to_torch_28(fixed_images),
                    os.path.join(out_dir, f"{batches_done}.png"),
                    nrow=7,
                )
                save_image(
                    denorm_to_torch_28(real_x),
                    os.path.join(out_dir, "real_samples.png"),
                    nrow=7,
                )
                saved_initial = True

            d_loss, wdist = critic_step(real_x)
            wasserstein_distance_history.append(float(wdist.numpy()))

            g_loss_val = None
            if (epoch * len(dataloader) + i_batch + 1) % n_critic == 0:
                g_loss_val = generator_step(bs)

                batches_done += n_critic
                np.save(
                    os.path.join(out_dir, "wasserstein_distance.npy"),
                    np.array(wasserstein_distance_history, dtype=np.float32),
                )

                if batches_done % sample_interval == 0:
                    fixed_images = generator(fixed_z[:bs], training=False)
                    save_image(
                        denorm_to_torch_28(fixed_images),
                        os.path.join(out_dir, f"{batches_done}.png"),
                        nrow=7,
                    )
                    critic.save_weights(
                        os.path.join(out_dir, f"critic-{batches_done}.weights.h5")
                    )
                    generator.save_weights(
                        os.path.join(out_dir, f"generator-{batches_done}.weights.h5")
                    )

            if g_loss_val is None:
                pbar.set_postfix(
                    {"D": float(d_loss.numpy()), "W": float(wdist.numpy())}
                )
            else:
                pbar.set_postfix(
                    {
                        "D": float(d_loss.numpy()),
                        "G": float(g_loss_val.numpy()),
                        "W": float(wdist.numpy()),
                    }
                )

    critic.save_weights(os.path.join(out_dir, "critic-final.weights.h5"))
    generator.save_weights(os.path.join(out_dir, "generator-final.weights.h5"))
    import json

    config = {
        "image_size": image_size,
        "channels": 1,
        "n_data_qubits": n_data_qubits,
        "n_ancillas": 0,
        "n_layers": n_layers,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "classes": classes,
        "prior": "calibrated",
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=str, default="0", help="e.g. '0' or '0,1,2'")
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument(
        "--n_data_qubits",
        type=int,
        default=10,
        help="Use 10 qubits, generator outputs 1024 then slice to 784 in model",
    )
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--out", type=str, default="./pqwgan_28x28_FullUnitary")
    parser.add_argument("--per_class_size", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train(
        classes_str=args.classes,
        n_layers=args.layers,
        n_data_qubits=args.n_data_qubits,
        batch_size=args.batch_size,
        out_dir=args.out,
        per_class_size=args.per_class_size,
        n_epochs=args.epochs,
    )
