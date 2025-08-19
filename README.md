# Transfer Learning for Diffusion Model simulated

In this repository, I replicated the methods from the paper [Transfer Learning for Diffusion Models](https://arxiv.org/abs/2405.16876). This implementation includes training, regularization techniques, and visualization of diffusion models for 2D synthetic data. The original GitHub from the paper can be found [here](https://github.com/YiDongOuYang/Transfer-Learning-for-Diffusion-Models).

1. **Clone the repository**:
   ```bash
   git clone git@github.com:juliankleutgens/TransferLearingForDM_simulated.git
   ```

2. **Navigate into the project directory**:
   ```bash
   cd TransferLearingForDM_simulated
   ```

3. **Set up the environment**:
   ```bash
   conda create --name tgdp python=3.9
   conda activate tgdp
   pip install torch torchvision numpy matplotlib torchsummary
   ```

   

## Script Arguments

The script accepts the following command-line arguments:
```

| Argument               | Type   | Default   | Description                                      |
|------------------------|--------|-----------|--------------------------------------------------|
| `--n`                 | `int`  | `1000`    | Number of target samples (10, 100, 1000).       |
| `--eta_cycle`         | `float`| `0.0`     | Weight for the cycle regularization loss.       |
| `--eta_consistency`   | `float`| `0.0`     | Weight for the consistency regularization loss. |
| `--show_baseline`     | `flag` | `False`   | Show the baseline plots (no value needed).      |
| `--show_density_ratio`| `flag` | `False`   | Show the density ratio plots (no value needed). |
| `--device`            | `str`  | `cpu`     | Device to use (`cpu`, `cuda`, or `mps`).        |

```



## Features

- Train diffusion models on 2D synthetic data.
- Regularization techniques such as cycle consistency and consistency loss.
- Visualization of baseline and density ratio plots.

```bash
python main.py --n 1000 --eta_cycle 0 --eta_consistency 0 --show_density_ratio --device cpu
```

