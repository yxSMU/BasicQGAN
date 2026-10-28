# train_pqwgan_tf_tc_prob16_emnist_letters.py

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import argparse
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms as torchvision_transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
from PIL import Image

import tensorcircuit as tc

tc.set_backend("tensorflow")

from models.size16_past_tc_analyze_PiorApplyPast_FullUnitary import PQWGAN_CC


class EMNISTImageDataset(Dataset):
    def __init__(self, root_dir, letter_idx, transform=None, per_class_size=1500):
        self.root_dir = root_dir
        self.letter_idx = letter_idx
        self.transform = transform

        letter_dir = os.path.join(root_dir, str(letter_idx))
        self.image_files = sorted(os.listdir(letter_dir))[:per_class_size]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.root_dir, str(self.letter_idx), self.image_files[idx]
        )
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, self.letter_idx


LETTER_COEFFICIENTS = {
    0: {  # A
        "a": 0.5201353430747986,
        "b": 0.002441821154206991,
        "c": -0.5327812433242798,
        "d": 0.5193394422531128,
        "e": 0.4946555197238922,
        "f": 0.4968266189098358,
        "g": -0.5500403046607971,
        "h": 0.5126915574073792,
    },
    1: {  # B
        "a": 0.522555947303772,
        "b": 0.570054829120636,
        "c": -0.4296203553676605,
        "d": 0.5339306592941284,
        "e": 0.5560710430145264,
        "f": 0.5628848075866699,
        "g": -0.587906539440155,
        "h": 0.5352874398231506,
    },
    2: {  # C
        "a": 0.5304654836654663,
        "b": -0.4840916395187378,
        "c": 0.15759608149528503,
        "d": 0.5211851596832275,
        "e": -0.504025936126709,
        "f": -0.5124938488006592,
        "g": 0.08970972150564194,
        "h": 0.5193774700164795,
    },
    3: {  # D
        "a": 0.5466839075088501,
        "b": 0.5821467041969299,
        "c": -0.09719526767730713,
        "d": 0.5496084690093994,
        "e": 0.5892109870910645,
        "f": 0.5725294351577759,
        "g": -0.5323576927185059,
        "h": 0.5315254926681519,
    },
    4: {  # E
        "a": 0.48038026690483093,
        "b": 0.516901969909668,
        "c": -0.4882996678352356,
        "d": 0.4745725989341736,
        "e": 0.3916677236557007,
        "f": 0.47446587681770325,
        "g": -0.4813341796398163,
        "h": 0.4827532470226288,
    },
    5: {  # F
        "a": 0.47698885202407837,
        "b": -0.36761680245399475,
        "c": 0.4203091859817505,
        "d": 0.47261595726013184,
        "e": -0.3603479862213135,
        "f": -0.37239184975624084,
        "g": 0.4295998811721802,
        "h": 0.4666997492313385,
    },
    6: {  # G
        "a": 0.5436503291130066,
        "b": 0.5625181794166565,
        "c": -0.20838043093681335,
        "d": 0.5496839880943298,
        "e": 0.5502952933311462,
        "f": 0.5523372888565063,
        "g": -0.5296730995178223,
        "h": 0.5466009378433228,
    },
    7: {  # H
        "a": 0.5484362244606018,
        "b": -0.5727034211158752,
        "c": -0.11357541382312775,
        "d": 0.5505297183990479,
        "e": -0.5408613085746765,
        "f": -0.5443267822265625,
        "g": -0.03089028410613537,
        "h": 0.547027587890625,
    },
    8: {  # I
        "a": -0.5103759765625,
        "b": 0.09703099727630615,
        "c": -0.037822261452674866,
        "d": -0.48423340916633606,
        "e": 0.12069525569677353,
        "f": 0.11269713193178177,
        "g": 0.04450569301843643,
        "h": -0.4578574001789093,
    },
    9: {  # J
        "a": 0.4880017936229706,
        "b": -0.4732731580734253,
        "c": 0.3500075340270996,
        "d": 0.4999585449695587,
        "e": -0.4610886573791504,
        "f": -0.45963266491889954,
        "g": 0.2876162528991699,
        "h": 0.484636515378952,
    },
}


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[Warn] No GPU found. Running on CPU.")
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[OK] Using GPU(s): {gpus}")


def denorm_to_torch_16(x_tf_flat: tf.Tensor) -> torch.Tensor:
    """
    [-1,1] -> [0,1]
    """
    x = tf.reshape(x_tf_flat, (-1, 1, 16, 16))
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(x, 0.0, 1.0)
    return torch.from_numpy(x.numpy())


def compute_gradient_penalty_tf(critic, real_x, fake_x):
    """ """
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
    letter_idx: int,
    n_layers: int,
    n_data_qubits: int,
    batch_size: int,
    out_dir: str,
    per_class_size: int = 1000,
    n_epochs: int = 50,
):
    setup_gpu()
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[Letter] Training on EMNIST letter: {chr(65 + letter_idx)} (idx={letter_idx})"
    )

    image_size = 16
    output_dim = image_size * image_size

    lr_D = 0.0002
    lr_G = 0.01
    b1, b2 = 0.0, 0.9
    lambda_gp = 10.0
    n_critic = 5
    sample_interval = 10

    n_ancillas = 0
    latent_dim = n_data_qubits

    coeffs = LETTER_COEFFICIENTS[letter_idx]
    a, b, c, d, e, f, g, h = (
        coeffs["a"],
        coeffs["b"],
        coeffs["c"],
        coeffs["d"],
        coeffs["e"],
        coeffs["f"],
        coeffs["g"],
        coeffs["h"],
    )

    base_scales = tf.constant([a, b, c, d, e, f, g, h], dtype=tf.float32) * np.pi

    def sample_z(bs: tf.Tensor) -> tf.Tensor:
        if latent_dim <= int(base_scales.shape[0]):
            scale = base_scales[:latent_dim]
        else:
            pad = (
                tf.ones([latent_dim - int(base_scales.shape[0])], dtype=tf.float32)
                * np.pi
            )
            scale = tf.concat([base_scales, pad], axis=0)

        z_rand = tf.random.uniform([bs, latent_dim], 0.0, 1.0, dtype=tf.float32)
        z = z_rand * tf.reshape(scale, [1, latent_dim])
        return z

    dataset = EMNISTImageDataset(
        root_dir="./emnist_uppercase_rotated_180_resize",
        letter_idx=letter_idx,
        transform=torchvision_transforms.Compose(
            [
                ToTensor(),
                torchvision_transforms.Lambda(lambda x: x.view(-1)),
            ]
        ),
        per_class_size=per_class_size,
    )

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
        n_data_qubits=n_data_qubits,
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
        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{n_epochs} ({chr(65 + letter_idx)})"
        )
        for i, (real_x_pt, _) in enumerate(pbar):
            real_x = tf.convert_to_tensor(real_x_pt.numpy(), dtype=tf.float32)
            real_x = real_x * 2.0 - 1.0  # [0,1] -> [-1,1]
            bs = tf.shape(real_x)[0]

            if not saved_initial:
                fixed_images = generator(fixed_z[:bs], training=False)
                save_image(
                    denorm_to_torch_16(fixed_images),
                    os.path.join(out_dir, f"{batches_done}.png"),
                    nrow=5,
                )
                save_image(
                    denorm_to_torch_16(real_x),
                    os.path.join(out_dir, "real_samples.png"),
                    nrow=5,
                )
                saved_initial = True

            d_loss, wdist = critic_step(real_x)
            wasserstein_distance_history.append(float(wdist.numpy()))

            g_loss_val = None
            if (epoch * len(dataloader) + i + 1) % n_critic == 0:
                g_loss_val = generator_step(bs)

                batches_done += n_critic
                np.save(
                    os.path.join(out_dir, "wasserstein_distance.npy"),
                    np.array(wasserstein_distance_history, dtype=np.float32),
                )

                if batches_done % sample_interval == 0:
                    fixed_images = generator(fixed_z[:bs], training=False)
                    save_image(
                        denorm_to_torch_16(fixed_images),
                        os.path.join(out_dir, f"{batches_done}.png"),
                        nrow=5,
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
        "n_ancillas": n_ancillas,
        "n_layers": n_layers,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "letter": chr(65 + letter_idx),
        "letter_idx": letter_idx,
        "coefficients": coeffs,
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    np.save(
        os.path.join(out_dir, "wasserstein_distance_final.npy"),
        np.array(wasserstein_distance_history, dtype=np.float32),
    )

    print("Done")
    print(f"Model weights saved to: {out_dir}")
    print(f"  - Letter: {chr(65 + letter_idx)} (idx={letter_idx})")
    print(f"  - critic-final.weights.h5")
    print(f"  - generator-final.weights.h5")
    print(f"  - model_config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--letter", type=int, default=0, help="Letter index: 0=A, 1=B, ..., 9=J"
    )
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument(
        "--n_data_qubits", type=int, default=8, help="16x16=256=2^8 -> set 8"
    )
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--out", type=str, default="./train_emnist_letters/A")
    parser.add_argument("--per_class_size", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train(
        letter_idx=args.letter,
        n_layers=args.layers,
        n_data_qubits=args.n_data_qubits,
        batch_size=args.batch_size,
        out_dir=args.out,
        per_class_size=args.per_class_size,
        n_epochs=args.epochs,
    )
