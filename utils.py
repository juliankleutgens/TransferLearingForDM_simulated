import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# Simplified function for plotting
def plot_data(data, title, labels=None):
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().numpy()
    plt.figure(figsize=(8, 6))
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.7, s=1)
        plt.title(title)
        plt.xlabel("X1")
        plt.ylabel("X2")
        #plt.xlim([-1.5, 1.5])
        #plt.ylim([-1.5, 1.5])
        plt.tight_layout()
        plt.show()
        return None

    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(data[idx, 0], data[idx, 1], label=f"Label {int(label)}", alpha=0.6)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_source_target_continuous(source_data, source_labels, target_data, target_labels, title):
    plt.figure(figsize=(10, 8))

    # Plot source data with a color spectrum
    scatter_source = plt.scatter(
        source_data[:, 0].numpy(),
        source_data[:, 1].numpy(),
        c=source_labels.numpy(),
        cmap="viridis",
        label="Source Domain",
        alpha=0.6
    )
    plt.colorbar(scatter_source, label="Source Label (Uncertainty)")

    # Plot target data with a separate color spectrum
    scatter_target = plt.scatter(
        target_data[:, 0].numpy(),
        target_data[:, 1].numpy(),
        c=target_labels.numpy(),
        cmap="plasma",
        label="Target Domain",
        alpha=0.6
    )
    plt.colorbar(scatter_target, label="Target Label (Uncertainty)")

    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to plot source and target data in one figure with 4 colors
def plot_source_target(source_data, source_labels, target_data, target_labels, title):
    plt.figure(figsize=(10, 8))

    # Plot source data
    for label in torch.unique(source_labels):
        idx = source_labels == label
        plt.scatter(
            source_data[idx, 0].numpy(),
            source_data[idx, 1].numpy(),
            label=f"Source Label {int(label)}",
            alpha=0.6
        )

    # Plot target data
    for label in torch.unique(target_labels):
        idx = target_labels == label
        plt.scatter(
            target_data[idx, 0].numpy(),
            target_data[idx, 1].numpy(),
            label=f"Target Label {int(label)}",
            alpha=0.6
        )

    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    plt.show()





def plot_noise_ScheduleVP():
    from main import NoiseScheduleVP
    nosie_sedule = NoiseScheduleVP(beta_0=0.1, beta_1=20.0)
    plt.figure(figsize=(8, 6))
    ts = torch.linspace(0, 1, 100)
    plt.plot(ts, nosie_sedule.marginal_alpha(ts), label="alpha_t")
    plt.plot(ts, nosie_sedule.marginal_std(ts), label="sigma_t")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_true_and_diffused(true_data, diffused_data, true_labels=None, diffused_labels=None, title_true_data="True Source Distribution", title_diffused_data="Diffused Target Distribution"):
    # Convert tensors to numpy arrays if needed
    if isinstance(true_data, torch.Tensor):
        true_data = true_data.detach().numpy()
    if isinstance(diffused_data, torch.Tensor):
        diffused_data = diffused_data.detach().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().numpy()
    if isinstance(diffused_labels, torch.Tensor):
        diffused_labels = diffused_labels.detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: True data distribution
    if true_labels is None:
        axes[0].scatter(true_data[:, 0], true_data[:, 1], alpha=0.7, s=1)
    else:
        for label in np.unique(true_labels):
            idx = true_labels == label
            axes[0].scatter(true_data[idx, 0], true_data[idx, 1], label=f"Label {int(label)}", alpha=0.6)
        axes[0].legend()
    axes[0].set_title(title_true_data)
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")
    axes[0].set_xlim([-3, 3])
    axes[0].set_ylim([-3, 3])
    axes[0].grid(True)

    # Right plot: Diffused data distribution
    if diffused_labels is None:
        axes[1].scatter(diffused_data[:, 0], diffused_data[:, 1], alpha=0.7, s=1)
    else:
        for label in np.unique(diffused_labels):
            idx = diffused_labels == label
            axes[1].scatter(diffused_data[idx, 0], diffused_data[idx, 1], label=f"Label {int(label)}", alpha=0.6)
        axes[1].legend()
    axes[1].set_title(title_diffused_data)
    axes[1].set_xlabel("X1")
    axes[1].set_ylabel("X2")
    axes[1].set_xlim([-3, 3])
    axes[1].set_ylim([-3, 3])
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_domain_classifier(model, source_data, target_data, device='cpu'):
    """
    Evaluate the domain classifier on source and target data
    """
    source_labels = torch.ones(source_data.shape[0])
    target_labels = torch.zeros(target_data.shape[0])

    dataset = TensorDataset(torch.cat([source_data, target_data]), torch.cat([source_labels, target_labels]))
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)

    model.eval()
    model.to(device)

    overall_fpr = 0
    overall_acc = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        y_pred = model(x)
        y = y.unsqueeze(1)
        acc = (y_pred > 0.5) == (y > 0.5)
        acc = acc.float().mean()
        overall_acc += acc.item()
        # False positive rate
        fpr = ((y_pred > 0.5) & (y < 0.5)).float().sum()
        overall_fpr += fpr.item()
    overall_acc /= len(loader)
    fpr = overall_fpr / target_data.shape[0]
    print(f"Domain Classifier Accuracy: {overall_acc:.4f}")
    print(f"Domain Classifier FPR: {fpr:.4f}")

def logistic_loss(x_source, y_target):
    b = x_source.size(0) + y_target.size(0)
    # c_w(x_p) - source domain
    log_prob_source = torch.log(x_source + 1e-8).mean()
    # 1 - c_w(x_q) - target domain
    log_prob_target = torch.log(1 - y_target + 1e-8).mean()
    return (-log_prob_source - log_prob_target) / b


class DPM_Solver_all_functions:
    """
    A minimal DPM-Solver for unconditional noise-based models.
    We show only a simple single-step solver (DPM-Solver-1) for demonstration.
    For higher-order methods, see the original code.
    """
    def __init__(self, model_fn, noise_schedule, eta=0.0, guidance_network=None):
        self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
        self.noise_schedule = noise_schedule
        self.algorithm_type = "dpmsolver"  # or "dpmsolver++"
        self.eta = 0.9  # extra noise
        self.guidance_network = guidance_network

    def dpm_solver_first_update(self, x, s, t):
        """
        DPM-Solver-1 (similar to DDIM) from time s to t:
        x_{t} = alpha_t/alpha_s * x_s - sigma_t * (exp(lambda_t - lambda_s) - 1) * model_s, etc.
        (Implementation can vary; this is just a minimal illustration.)
        """
        ns = self.noise_schedule
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_s = torch.exp(log_alpha_s)
        alpha_t = torch.exp(log_alpha_t)
        sigma_s = ns.marginal_std(s)
        sigma_t = ns.marginal_std(t)
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        phi_1 = torch.expm1(h)

        # Evaluate model at s
        model_s = self.model(x, s)  # predicted noise
        # Approx update: x_t
        x_t = (torch.exp(log_alpha_t - log_alpha_s) * x - (sigma_t * phi_1) * model_s)

        # Inject extra noise
        if self.eta > 0:
            # Simple variance scaling for the extra noise
            diff_var = (sigma_t**2 - sigma_s**2).clamp(min=0)
            noise_scale = self.eta * torch.sqrt(diff_var)
            x_t = x_t + noise_scale * torch.randn_like(x)

        return x_t


    def first_order_sample(self, x, steps=10, t_start=1.0, t_end=1e-3):
        """
        Minimal single-step approach where we just linearly space the times.
        """
        device = x.device
        ts = torch.linspace(t_start, t_end, steps+1, device=device)
        # Go from s=ts[i], t=ts[i+1].
        for i in range(steps):
            s = ts[i].view(1)
            t = ts[i+1].view(1)
            x = self.dpm_solver_first_update(x, s, t)
        return x
    def second_order_update(self, x, s, t):
        """
        Singlestep second‐order update from DPM‐Solver code (dpmsolver variant).
        We pick r1=0.5 for simplicity.
        """
        r1 = 0.5
        ns = self.noise_schedule
        lam_s = ns.marginal_lambda(s)
        lam_t = ns.marginal_lambda(t)
        h = lam_t - lam_s
        lam_s1 = lam_s + r1 * h
        s1 = ns.inverse_lambda(lam_s1)

        # Evaluate model at s if needed
        model_s = self.model(x, s)  # predicted noise

        # Step to s1
        alpha_s1 = torch.exp(ns.marginal_log_mean_coeff(s1))
        sigma_s1 = ns.marginal_std(s1)
        phi_11 = torch.expm1(r1*h)
        x_s1 = (torch.exp(ns.marginal_log_mean_coeff(s1) - ns.marginal_log_mean_coeff(s)) * x
                - sigma_s1 * phi_11 * model_s)

        # Evaluate model at s1
        model_s1 = self.model(x_s1, s1)

        # Final step to t
        alpha_t = torch.exp(ns.marginal_log_mean_coeff(t))
        sigma_t = ns.marginal_std(t)
        phi_1 = torch.expm1(h)
        x_t = ((torch.exp(ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(s))) * x
               - sigma_t * phi_1 * model_s - 0.5 * (sigma_t * phi_1) * (model_s1 - model_s))  # second‐order correction

        return x_t



    def second_order_sample(self, x, steps=10, t_start=1.0, t_end=1e-3):
        """
        Start from x at t=1.0 (standard normal), go down to t=1e-3,
        using second‐order steps. This uses a simple 'multistep' approach.
        """
        device = x.device
        ts = torch.linspace(t_start, t_end, steps+1).to(device)
        # Initialize with a single first_order_update
        s = ts[0].unsqueeze(0)
        for step in range(1, steps+1):
            t = ts[step].unsqueeze(0)
            if step == 1:
                x = self.dpm_solver_first_update(x, s, t)
            else:
                x = self.second_order_update(x, s, t)
            s = t
        return x


    def compute_guidance_noise(self, x, t_, guidance_network, noise_schedule):
        """
        Returns the guidance 'noise' that, when added to the source-model's predicted noise,
        effectively adds ∇_x log h_psi(x,t_) to the total score.

        guidance_noise = - sigma_t * ∇_x log h_psi(x,t_).
        """
        # 1. Require gradient on x
        x = x.detach()
        x.requires_grad_(True)

        # 2. Evaluate h_psi(x, t_) (scalar per sample)
        t_ = t_.unsqueeze(1).expand(x.shape[0], -1)
        h_val = guidance_network(torch.cat([x, t_], dim=1))  # shape [batch,1]

        # 3. sum(log(...)) so that grad(...) w.r.t. x is the sum of ∇ log(h_val_i)
        log_sum = torch.log(h_val + 1e-20).sum()
        grad_log_h = torch.autograd.grad(log_sum, x, create_graph=False)[0]  # shape [batch, d]

        # 4. Convert that gradient to "noise" by multiplying by -sigma_t
        sigma_t = noise_schedule.marginal_std(t_)
        guidance_noise = -sigma_t * grad_log_h

        x.requires_grad_(False)
        return guidance_noise.detach()


    def second_order_update_guidance(self, x, s, t):
        """
        Single-step second-order update from DPM-Solver,
        now with an added guidance gradient.

        We pick r1=0.5 for simplicity.
        """
        r1 = 0.5
        scale1 = 1
        scale2 = 1
        ns = self.noise_schedule
        lam_s = ns.marginal_lambda(s)
        lam_t = ns.marginal_lambda(t)
        h = lam_t - lam_s
        lam_s1 = lam_s + r1 * h
        s1 = ns.inverse_lambda(lam_s1)

        model_s_source = self.model(x, s)  # shape [batch, d], source model’s predicted noise
        guidance_s = self.compute_guidance_noise(x, s, self.guidance_network, ns)
        model_s_combined = model_s_source + scale1*guidance_s

        # Step to intermediate s1
        alpha_s1 = torch.exp(ns.marginal_log_mean_coeff(s1))
        sigma_s1 = ns.marginal_std(s1)
        phi_11 = torch.expm1(r1 * h)
        x_s1 = (
                torch.exp(ns.marginal_log_mean_coeff(s1) - ns.marginal_log_mean_coeff(s)) * x
                - sigma_s1 * phi_11 * model_s_combined)

        #
        model_s1_source = self.model(x_s1, s1)
        guidance_s1 = self.compute_guidance_noise(x_s1, s1, self.guidance_network, ns)
        model_s1_combined = model_s1_source + scale2*guidance_s1

        # Final step to t (2nd-order update)
        alpha_t = torch.exp(ns.marginal_log_mean_coeff(t))
        sigma_t = ns.marginal_std(t)
        phi_1 = torch.expm1(h)

        x_t = (
                torch.exp(ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(s)) * x
                - sigma_t * phi_1 * model_s_combined
                - 0.5 * (sigma_t * phi_1) * (model_s1_combined - model_s_combined)
        )

        return x_t

    def first_order_update_guidance(self, x, s, t):
        """
        Single-step first-order update from DPM-Solver,
        with guidance gradient.
        """
        ns = self.noise_schedule
        lam_s = ns.marginal_lambda(s)
        lam_t = ns.marginal_lambda(t)
        h = lam_t - lam_s

        model_s_source = self.model(x, s)  # shape [batch, d], source model’s predicted noise
        guidance_s = self.compute_guidance_noise(x, s, self.guidance_network, ns)
        model_s_combined = model_s_source + guidance_s

        # Final step to t (1st-order update)
        alpha_t = torch.exp(ns.marginal_log_mean_coeff(t))
        sigma_t = ns.marginal_std(t)
        phi_1 = torch.expm1(h)

        x_t = (
                torch.exp(ns.marginal_log_mean_coeff(t) - ns.marginal_log_mean_coeff(s)) * x
                - sigma_t * phi_1 * model_s_combined
        )

        return x_t


    def second_order_sample_guidance(self, x, steps=10, t_start=1.0, t_end=1e-3):
        """
        Start from x at t=1.0 (standard normal), go down to t=1e-3,
        using second‐order steps. This uses a simple 'multistep' approach.
        """
        device = x.device
        ts = torch.linspace(t_start, t_end, steps+1).to(device)
        # Initialize with a single first_order_update
        s = ts[0].unsqueeze(0)
        for step in range(1, steps+1):
            t = ts[step].unsqueeze(0)
            if step == 1:
                x = self.first_order_update_guidance(x, s, t)
            else:
                x = self.second_order_update_guidance(x, s, t)
            s = t
        return x



import math
import torch

def gaussian_pdf(x, mean, sigma2):
    """
    x: Tensor of shape [N, d]  (e.g. d=2)
    mean: Tensor of shape [d]
    sigma2: float
    Returns the pdf value N(x | mean, sigma2 I) for each row of x.
    """
    d = x.shape[1]
    # (x - mean)
    diff = x - mean
    # exponent term = -(1/(2*sigma2))*||x - mean||^2
    exp_term = -0.5 / sigma2 * (diff * diff).sum(dim=1)  # shape [N]
    # normalizing constant = 1 / ( (2*pi*sigma2)^(d/2) )
    denom = (2.0 * math.pi * sigma2) ** (d / 2.0)
    # pdf = 1/denom * exp(exp_term)
    pdf_val = torch.exp(exp_term) / denom
    return pdf_val

def mixture_pdf(x, mu, sigma2):
    """
    Computes q(x) = 0.5*N(x|mu_T, sigma2 I) + 0.5*N(x|-mu_T, sigma2 I).
    x: [N, d]
    mu_T: [d]
    sigma2: float
    """
    pdf_pos = gaussian_pdf(x,  mu,  sigma2)  # N(x | +mu_T, sigma2 I)
    pdf_neg = gaussian_pdf(x, -mu,  sigma2)  # N(x | -mu_T, sigma2 I)
    return 0.5 * pdf_pos + 0.5 * pdf_neg


def estimate_ratio(x, classifier):
    """
    Given a batch of points x, compute the learned ratio r(x) = (1 - c_ω(x)) / c_ω(x).
    x: Tensor [N, 2]
    classifier: your trained binary classifier (source vs. target)
    """
    with torch.no_grad():
        cvals = classifier(x)  # cvals in [0,1], shape [N,1]
    return (1 - cvals) / (cvals + 1e-20)  # Avoid dividing by 0

def true_ratio_oracle(x, muS, muT, sigma2):
    """
    If you know the true p and q are two-component Gaussians,
    then r(x) = q(x)/p(x). This is your 'oracle' ratio.
    """
    q_vals = mixture_pdf(x, muT, sigma2)
    p_vals = mixture_pdf(x, muS, sigma2)
    return q_vals / p_vals



def plot_ration(x_rand, log_ratio_oracle, log_ratio_learned):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Left: Oracle
    sc1 = axes[0].scatter(
        x_rand[:, 0].numpy(),
        x_rand[:, 1].numpy(),
        c=log_ratio_oracle.numpy(),
        cmap='viridis',
        s=20
    )
    axes[0].set_title("(a) Oracle")
    axes[0].set_xlim([-1.5, 1.5])
    axes[0].set_ylim([-1.5, 1.5])
    cb1 = plt.colorbar(sc1, ax=axes[0])
    cb1.set_label("log ratio")

    # Right: Learned
    sc2 = axes[1].scatter(
        x_rand[:, 0].numpy(),
        x_rand[:, 1].numpy(),
        c=log_ratio_learned.numpy(),
        cmap='viridis',
        s=20
    )
    axes[1].set_title("(b) Learned ratio")
    axes[1].set_xlim([-1.5, 1.5])
    axes[1].set_ylim([-1.5, 1.5])
    cb2 = plt.colorbar(sc2, ax=axes[1])
    cb2.set_label("log ratio")

    plt.tight_layout()
    plt.show()
