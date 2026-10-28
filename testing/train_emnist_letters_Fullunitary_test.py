# train_emnist_letters_Fullunitary_test.py

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import argparse
import json
import tensorflow as tf
import torch
import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorcircuit as tc

tc.set_backend("tensorflow")

from torchvision.transforms import ToTensor
from torchvision import transforms as torchvision_transforms
from models.size16_past_tc_analyze_PiorApplyPast_FullUnitary import PQWGAN_CC


class EMNISTImageDataset(Dataset):
    def __init__(self, root_dir, letter_idx, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.letter_idx = letter_idx
        self.transform = transform

        letter_dir = os.path.join(root_dir, str(letter_idx))
        self.image_files = sorted(os.listdir(letter_dir))
        if max_samples:
            self.image_files = self.image_files[:max_samples]

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


def prepare_images_for_metrics(real_images_np, fake_images_np, target_size=(28, 28)):
    """ """
    real_resized = []
    fake_resized = []

    for real_np, fake_np in zip(real_images_np, fake_images_np):
        real_img = Image.fromarray((real_np * 255).astype(np.uint8)).convert("L")
        fake_img = Image.fromarray((fake_np * 255).astype(np.uint8)).convert("L")

        real_img = real_img.resize(target_size, Image.LANCZOS)
        fake_img = fake_img.resize(target_size, Image.LANCZOS)

        real_array = np.array(real_img).astype(np.float32) / 255.0
        fake_array = np.array(fake_img).astype(np.float32) / 255.0

        real_resized.append(real_array)
        fake_resized.append(fake_array)

    return np.array(real_resized), np.array(fake_resized)


def calculate_fid(real_features, fake_features):
    """ """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    epsilon = 1e-6
    sigma1 += epsilon * np.eye(sigma1.shape[0])
    sigma2 += epsilon * np.eye(sigma2.shape[0])

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_ssim_psnr(real_images, fake_images):
    """ """
    ssim_scores = []
    psnr_scores = []

    for real_array, fake_array in zip(real_images, fake_images):
        mu1 = real_array.mean()
        mu2 = fake_array.mean()

        sigma1 = ((real_array - mu1) ** 2).mean()
        sigma2 = ((fake_array - mu2) ** 2).mean()
        sigma12 = ((real_array - mu1) * (fake_array - mu2)).mean()

        c1 = 0.01**2
        c2 = 0.03**2

        luminance = (2 * mu1 * mu2 + c1) / (mu1**2 + mu2**2 + c1)
        contrast = (2 * np.sqrt(sigma1) * np.sqrt(sigma2) + c2) / (sigma1 + sigma2 + c2)
        structure = (sigma12 + c2 / 2) / (np.sqrt(sigma1) * np.sqrt(sigma2) + c2 / 2)

        ssim_value = luminance * contrast * structure
        ssim_scores.append(ssim_value)

        psnr_value = 10 * np.log10(
            1.0**2 / ((real_array - fake_array) ** 2).mean() + 1e-10
        )
        psnr_scores.append(psnr_value)

    return (
        np.mean(ssim_scores),
        np.std(ssim_scores),
        np.mean(psnr_scores),
        np.std(psnr_scores),
    )


def test(config_path: str, out_dir: str, n_images: int = 250):
    setup_gpu()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "fake"), exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"[Config] {config}")

    image_size = config.get("image_size", 16)
    n_data_qubits = config.get("n_data_qubits", 8)
    n_layers = config.get("n_layers", 40)
    latent_dim = config.get("latent_dim", 8)
    letter_idx = config.get("letter_idx", 0)
    coefficients = config.get("coefficients", {})

    gan = PQWGAN_CC(
        image_size=image_size,
        channels=1,
        n_data_qubits=n_data_qubits,
        n_layers=n_layers,
        use_born_prob=True,
    )
    critic = gan.critic
    generator = gan.generator

    output_dim = image_size * image_size
    _ = critic(tf.zeros((1, output_dim), dtype=tf.float32))
    _ = generator(tf.zeros((1, latent_dim), dtype=tf.float32), training=False)

    generator.load_weights(
        os.path.join(os.path.dirname(config_path), "generator-final.weights.h5")
    )
    print(
        f"[Load] Generator weights loaded from {os.path.join(os.path.dirname(config_path), 'generator-final.weights.h5')}"
    )

    a = coefficients.get("a", 0.5)
    b = coefficients.get("b", 0.5)
    c = coefficients.get("c", -0.5)
    d = coefficients.get("d", 0.5)
    e = coefficients.get("e", 0.5)
    f = coefficients.get("f", 0.5)
    g = coefficients.get("g", -0.5)
    h = coefficients.get("h", 0.5)

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
        max_samples=n_images,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    real_images_list = []
    fake_images_list = []

    print(f"Generating {n_images} images...")
    pbar = tqdm(range(n_images), desc="Generating")
    for i in pbar:
        try:
            real_x_pt, _ = next(iter(dataloader))
        except StopIteration:
            break

        real_x = tf.convert_to_tensor(real_x_pt.numpy(), dtype=tf.float32)
        real_x = real_x * 2.0 - 1.0  # [0,1] -> [-1,1]

        z = sample_z(tf.constant(1, dtype=tf.int32))
        fake_x = generator(z, training=False)

        real_img = denorm_to_torch_16(real_x)
        fake_img = denorm_to_torch_16(fake_x)

        real_np = real_img.squeeze(0).squeeze(0).numpy()
        fake_np = fake_img.squeeze(0).squeeze(0).numpy()

        real_images_list.append(real_np)
        fake_images_list.append(fake_np)

        Image.fromarray((real_np * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "real", f"real_{i:04d}.png")
        )
        Image.fromarray((fake_np * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "fake", f"fake_{i:04d}.png")
        )

    real_images = np.array(real_images_list).astype(np.float32)
    fake_images = np.array(fake_images_list).astype(np.float32)

    print(f"Real images shape: {real_images.shape}")
    print(f"Fake images shape: {fake_images.shape}")

    print("\nPreparing images (resize to 28x28, normalize to [0,1])...")
    real_images_28, fake_images_28 = prepare_images_for_metrics(
        real_images, fake_images
    )
    print(f"Resized images shape: {real_images_28.shape}")

    print("\nCalculating FID...")
    real_features = real_images_28.reshape(real_images_28.shape[0], -1)
    fake_features = fake_images_28.reshape(fake_images_28.shape[0], -1)

    fid = calculate_fid(real_features, fake_features)
    print(f"FID: {fid:.2f}")

    print("Calculating SSIM and PSNR...")
    ssim_mean, ssim_std, psnr_mean, psnr_std = calculate_ssim_psnr(
        real_images_28, fake_images_28
    )
    print(f"SSIM: {ssim_mean:.4f} (+/- {ssim_std:.4f})")
    print(f"PSNR: {psnr_mean:.2f} (+/- {psnr_std:.2f})")

    print("\n" + "=" * 50)
    print("Results:")
    print(f"FID: {fid:.2f}")
    print(f"SSIM: {ssim_mean:.4f} (+/- {ssim_std:.4f})")
    print(f"PSNR: {psnr_mean:.2f} (+/- {psnr_std:.2f})")
    print("=" * 50)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(real_images[0], cmap="gray")
    plt.title("Real Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(fake_images[0], cmap="gray")
    plt.title("Fake Image")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_comparison.png"), dpi=150)
    plt.close()
    print(
        f"[Save] Sample comparison saved to {os.path.join(out_dir, 'sample_comparison.png')}"
    )

    results = {
        "fid": float(fid),
        "ssim_mean": float(ssim_mean),
        "ssim_std": float(ssim_std),
        "psnr_mean": float(psnr_mean),
        "psnr_std": float(psnr_std),
        "n_images": n_images,
    }
    np.save(os.path.join(out_dir, "metrics.npy"), results)
    print(f"[Save] Results saved to {os.path.join(out_dir, 'metrics.npy')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model_config.json"
    )
    parser.add_argument(
        "--n_images", type=int, default=250, help="Number of images for testing"
    )
    parser.add_argument(
        "--out", type=str, default="./test_emnist_fullunitary", help="Output directory"
    )
    args = parser.parse_args()

    test(
        config_path=args.config,
        out_dir=args.out,
        n_images=args.n_images,
    )
