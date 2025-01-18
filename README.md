# TransferLearningForDM_simulated

In this repository, I replicated the results from the paper **Transfer Learning for Diffusion Models**. This implementation includes training, regularization techniques, and visualization of diffusion models for 2D synthetic data.

## Environment Setup

To set up the required environment, follow these steps:

1. **Create a conda environment**:
   ```bash
   conda create --name tgdp python=3.9
   ```

2. **Activate the environment**:
   ```bash
   conda activate tgdp
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib torchsummary
   ```

## Script Arguments

The script accepts the following command-line arguments:

| Argument               | Type   | Default   | Description                                      |
|------------------------|--------|-----------|--------------------------------------------------|
| `--n`                 | `int`  | `1000`    | Number of target samples (10, 100, 1000).       |
| `--eta_cycle`         | `float`| `0.0`     | Weight for the cycle regularization loss.       |
| `--eta_consistency`   | `float`| `0.0`     | Weight for the consistency regularization loss. |
| `--show_baseline`     | `flag` | `False`   | Show the baseline plots (no value needed).      |
| `--show_density_ratio`| `flag` | `False`   | Show the density ratio plots (no value needed). |
| `--device`            | `str`  | `cpu`     | Device to use (`cpu`, `cuda`, or `mps`).        |

## Example Usage

Run the script with default settings:
```bash
python main.py
```

Run the script with custom arguments:
```bash
python main.py --n 1000  --show_density_ratio --device mps
```

## Features

- Train diffusion models on 2D synthetic data.
- Regularization techniques such as cycle consistency and consistency loss.
- Visualization of baseline and density ratio plots.

Feel free to contribute or create an issue if you encounter any problems!
```
