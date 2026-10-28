# BasicQGAN

Code for **Quantum Fidelity Landscape-Guided Prior Calibration for Single-Circuit QGAN Image Generation** ([paper](https://arxiv.org/html/2609.36702v1)). This project uses fixed prior ranges; the offline QFL calibration procedure is outside this code package.

## Setup

Use Python 3.10 with TensorFlow 2.17, TensorCircuit 0.12, PyTorch 2.14, torchvision 0.29, NumPy, SciPy, Pillow, Matplotlib, and tqdm. MNIST downloads automatically. Run commands from this project directory; place outputs under `outputs/`.

## Find an experiment

```bash
python run.py list
python run.py mnist16_main_noancilla --help
```

`run.py` resolves experiment names in `experiment_manifest.json`. Arguments after the name override preset values.

## Main experiment

Train each digit separately with the fixed calibrated prior coefficients in `training/train16size_Fullunitary_0.py`:

```bash
python run.py mnist16_main_noancilla --classes 0 --layers 40 --n_data_qubits 8 --per_class_size 1500 --batch_size 5 --epochs 50 --out outputs/mnist16_digit0
python run.py mnist16_eval_noancilla --config outputs/mnist16_digit0/model_config.json --n_images 250 --out outputs/mnist16_digit0_eval
```

For 28×28 images use `mnist28_main_noancilla` and `mnist28_eval_noancilla`, with 10 data qubits. Uppercase Letters uses `letters16_main_noancilla` and `letters16_eval_noancilla` and requires its image dataset in the format expected by the loader.

## Change parameters

| Argument | Meaning | Main setting |
| --- | --- | --- |
| `--classes` | MNIST class ID; run each class separately | `0` to `9` |
| `--layers` | Generator circuit depth | `40` |
| `--n_data_qubits` | Data qubits | `8` for 16×16; `10` for 28×28 |
| `--per_class_size` | Training samples per class | `1500` |
| `--batch_size`, `--epochs` | Training schedule | `5`, `50` |
| `--prior` | Fixed prior choice for the 16×16 model | `calibrated` |
| `--out` | Output directory | user chosen |

See [Experiment Guide](docs/experiments.md) for prior ablations, ancilla ablation, circuit-depth settings, and the PQWGAN(Global) baseline. Evaluators retain the original projects' pixel-vector FID/KID computation.
