# Experiment Guide

Run each command from the project root. Use a separate output directory for every setting. Pass each training run's `model_config.json` to its corresponding evaluator.

| Paper experiment | Training entry | Evaluation entry | Change |
| --- | --- | --- | --- |
| MNIST 16×16, no ancilla | `mnist16_main_noancilla --classes N` | `mnist16_eval_noancilla` | Main model, one digit per run |
| MNIST 28×28, no ancilla | `mnist28_main_noancilla --classes N` | `mnist28_eval_noancilla` | Image resolution and 10 data qubits |
| Uppercase Letters 16×16 | `letters16_main_noancilla` | `letters16_eval_noancilla` | Dataset and fixed letter prior |
| Digit 0, uniform `U(0,π)` prior | `mnist16_prior_uniform_pi` | `mnist16_eval_noancilla` | `--prior uniform_pi` |
| Digit 0, uniform `U(0,1)` prior | `mnist16_prior_uniform_1` | `mnist16_eval_noancilla` | `--prior uniform_1` |
| Digit 0, narrow prior | `mnist16_prior_narrow` | `mnist16_eval_noancilla` | `--prior narrow` |
| 16×16 with ancilla | `mnist16_ancilla_ablation --classes N` | `mnist16_eval_ancilla` | Add one ancilla initialized to zero |
| Circuit depth | `mnist16_main_noancilla --layers N` | `mnist16_eval_noancilla` | Set `N` to 10, 20, 40, 60, or 80 |
| PQWGAN(Global), 16×16 | `pqwgan_global_16 --classes N` | `pqwgan_global_16_eval` | Separate one-patch baseline; 8 data qubits and one ancilla |

Prior example:

```bash
python run.py mnist16_prior_uniform_pi --layers 40 --per_class_size 1500 --epochs 50 --out outputs/digit0_uniform_pi
python run.py mnist16_eval_noancilla --config outputs/digit0_uniform_pi/model_config.json --n_images 250 --out outputs/digit0_uniform_pi_eval
```

Baseline example:

```bash
python run.py pqwgan_global_16 --classes 0 --layers 20 --per_class_size 1500 --epochs 50 --out outputs/pqwgan_digit0
python run.py pqwgan_global_16_eval --config outputs/pqwgan_digit0/model_config.json --n_images 250 --out outputs/pqwgan_digit0_eval
```

The calibrated prior uses the fixed coefficients embedded in the training scripts. Change coefficients only when deliberately defining a new prior experiment. The evaluators use flattened image pixels; label results as **pixel-space FID/KID**. The Geometric Shapes dataset and offline QFL calibration are not included.
