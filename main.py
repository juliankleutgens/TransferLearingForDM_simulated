import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from itertools import cycle
import torchsummary


parser = argparse.ArgumentParser(description="Train your diffusion model.")
parser.add_argument("--n", type=int, default=1000, help="Number of target samples 10, 100, 1000")
parser.add_argument("--eta_cycle", type=float, default=0.0, help="Batch size")
parser.add_argument("--eta_consistency", type=float, default=0.0, help="Number of epochs")
parser.add_argument("--show_baseline", action="store_true", help="Show the baseline plots")
parser.add_argument("--show_density_ratio", action="store_true", help="Show the density ratio plots")
parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda or mps)")
args = parser.parse_args()

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Define constants
d = 2
sigma2 = 0.1
mu_S = torch.tensor([0.5, 0.5], dtype=torch.float32)
mu_T = torch.tensor([0.5, -0.5], dtype=torch.float32)
assert torch.dot(mu_S, mu_T) == 0
m = 10000 # Number of source samples
n = args.n # Number of target samples 10, 100, 1000


diffusion_steps = 25
device = args.device
# Training parameters
beta_0 = 0.1
beta_1 = 20.0
learning_rate = 1e-4
batch_size = 4096
# epochs
c_more_epochs = 5
vanilla_diffusion_epochs = 50*c_more_epochs
finetune_diffusion_epochs = 50*c_more_epochs
guidance_epochs = 20*c_more_epochs
source_epochs = 100*c_more_epochs

# Sampling
guidance_scale = [1, 1]
show_baseline = args.show_baseline
# Eta1 is the Cycle Loss and Eta2 is the Consistency Loss
# eta1and2 = [eta1, eta2] = [0, 0] for no regularization
eta1and2 = [args.eta_cycle, args.eta_consistency]
use_regularization = True if sum(eta1and2) > 0 else False
show_density_ratio = args.show_density_ratio

# -----------------------------------------------------------------------
# 1. Generate samples from source and target domains
# -----------------------------------------------------------------------
def generate_samples(mu, sigma2, num_samples):
    dist1 = torch.distributions.MultivariateNormal(mu, sigma2 * torch.eye(d))
    dist2 = torch.distributions.MultivariateNormal(-mu, sigma2 * torch.eye(d))
    samples1 = dist1.sample((num_samples // 2,))
    samples2 = dist2.sample((num_samples // 2,))
    labels = torch.cat([torch.ones(num_samples // 2), -torch.ones(num_samples // 2)])
    return torch.cat([samples1, samples2]), labels

source_data, source_labels = generate_samples(mu_S, sigma2, m)
target_data_all_n, target_labels_all_n = [], []
n_samples = [10, 100, 1000]
for n in n_samples:
    target_data, target_labels = generate_samples(mu_T, sigma2, n)
    target_data_all_n.append(target_data)
    target_labels_all_n.append(target_labels)
#plot_source_target(source_data, source_labels, target_data, target_labels, "Source and Target Distributions")
n_target_index = n_samples.index(n)
target_labels_ = target_labels_all_n[n_target_index]
target_data_ = target_data_all_n[n_target_index]

# -----------------------------------------------------------------------
# 2. Define the Networks: Noise Predictor, Guidance Network and Classifier
# -----------------------------------------------------------------------

class NoisePredictor(nn.Module):
    """
    A small MLP that takes (x_t, t) and predicts noise.
    We feed t as an extra input dimension.
    """
    def __init__(self, input_dim=d, hidden_dim=[512,512,512,512,256]):
        super().__init__()
        output_dim = input_dim
        in_dim = input_dim + 1  # +1 for t
        layers = []
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, t):
        # t is [batch], expand to [batch,1] then cat
        #t_emb = t.view(-1, 1).expand(-1, x.shape[1])
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        inp = torch.cat([x, t], dim=1)
        return self.model(inp)

#noise_predictor = NoisePredictor()
#torchsummary.summary(noise_predictor, input_size=[(2,),(1,)])


class GuidanceNetwork(nn.Module):
    """
    4-layer MLP with 512 hidden units and SiLU activation function
    Only x1 and x2 as input dimensions
    """
    def __init__(self, input_dim=d, hidden_dim=[512,512,512,512]):
        super().__init__()
        in_dim = input_dim+1 # +1 for t
        layers = []
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
#guidance_network = GuidanceNetwork()
#torchsummary.summary(guidance_network, input_size=[(2,)])

class Classifier(nn.Module):
    """
    The is no Information on how the classifier should look like in the Paper, so I defined it like the Guidance Network
    4-layer MLP with 512 hidden units and SiLU activation function and a sigmoid output layer
    Only x1 and x2 as input dimensions
    """
    def __init__(self, input_dim=d, hidden_dim=[512,512,512,512]):
        super().__init__()
        in_dim = input_dim
        layers = []
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
#classifier = Classifier()
#torchsummary.summary(classifier, input_size=[(2,)])

# -----------------------------------------------------------------------
# 3. Define the Noise Schedule from the Paper DPM Solver
# -----------------------------------------------------------------------
class NoiseScheduleVP:
    """
    Minimal version that lets us get alpha_t, sigma_t, etc.
    For simplicity, let's do a 'linear' schedule (as an example).
    """
    def __init__(self, beta_0=0.1, beta_1=20., dtype=torch.float32):
        self.schedule = 'linear'
        self.T = 1.0
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.dtype = dtype

    def marginal_log_mean_coeff(self, t):
        # log(alpha_t)
        # I asumed that they made a mistake in the paper and the formula should be like this
        # For linear VPSDE: alpha_t = exp(-0.5 * (beta_0 * t + 0.5*(beta_1-beta_0)*t^2))
        return -0.5*(self.beta_0*t + 0.5*(self.beta_1-self.beta_0)*t**2)

    def marginal_lambda(self, t):
        # Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].

        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2.*self.marginal_log_mean_coeff(t)))

    def inverse_lambda(self, lamb):
        # Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
        Delta = self.beta_0**2 + tmp
        return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)


noise_schedule = NoiseScheduleVP(beta_0=0.1, beta_1=20.0)

# -----------------------------------------------------------------------
# 3. Train the Noise Predictor - Based on the work: https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py#L30
# -----------------------------------------------------------------------
def train_diffusion_noise_prediction(
    model, data, n_epochs=5, batch_size=512, lr=1e-4, device='cpu'
):
    """
    Minimal training loop for noise-prediction on 2D data.
    q(x_t|x_0) ~ alpha_t * x_0 + sigma_t * eps, with t ~ Uniform(0,1).
    Loss = MSE(predicted_noise, true_noise).
    """
    if batch_size > data.size(0):
        batch_size = data.size(0)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    model.to(device)

    for epoch in range(n_epochs):
        for batch in loader:
            x0 = batch[0].to(device)    # shape [batch, 2]
            t_ = torch.rand(x0.size(0), device=device)  # uniform in [0,1]
            a_t = noise_schedule.marginal_alpha(t_).unsqueeze(1)
            s_t = noise_schedule.marginal_std(t_).unsqueeze(1)

            eps = torch.randn_like(x0)
            x_t = a_t * x0 + s_t * eps  # forward diffusion

            # Predict the noise
            eps_pred = model(x_t, t_)

            loss = mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Train Diffusion] epoch {epoch+1}/{n_epochs}, loss={loss.item():.4f}")

    model.to('cpu')



# -----------------------------------------------------------------------
# 4. Sample from the Noise Predictor and DPM-Solver: https://github.com/LuChengTHU/dpm-solver
# -----------------------------------------------------------------------
class DPM_Solver: # richtig
    """
    A minimal DPM-Solver for unconditional noise-based models.
    We show only a simple single-step solver (DPM-Solver-1) for demonstration.
    For higher-order methods, see the original code.
    """
    def __init__(self, model_fn, noise_schedule, guidance_scale=[0,0], guidance_network=None):
        self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
        self.noise_schedule = noise_schedule
        self.algorithm_type = "dpmsolver"  # or "dpmsolver++"
        self.eta = 0.9  # extra noise
        self.guidance_network = guidance_network
        self.scale1 = guidance_scale[0]
        self.scale2 = guidance_scale[1]

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

        r1: A float. The hyperparameter of the second-order solver.
        In the second-order update, we take a step of size r1 from s to s1.
        Then we take a step of s1 to t.
        """
        r1 = 0.5 # In the Pseudo Code they use 0.5 (Algorithm 1 DPM-Solver-2.) on page 6 of the Paper
        scale1 = self.scale1
        scale2 = self.scale2
        ns = self.noise_schedule
        lam_s = ns.marginal_lambda(s)
        lam_t = ns.marginal_lambda(t)
        h = lam_t - lam_s
        lam_s1 = lam_s + r1 * h
        s1 = ns.inverse_lambda(lam_s1)

        model_s_source = self.model(x, s)  # shape [batch, d], source model’s predicted noise
        guidance_s = self.compute_guidance_noise(x, s, self.guidance_network, ns) if self.guidance_network is not None else model_s_source
        model_s_combined = model_s_source + scale1 * guidance_s

        # Step to intermediate s1
        alpha_s1 = torch.exp(ns.marginal_log_mean_coeff(s1))
        sigma_s1 = ns.marginal_std(s1)
        phi_11 = torch.expm1(r1 * h)
        x_s1 = (
                torch.exp(ns.marginal_log_mean_coeff(s1) - ns.marginal_log_mean_coeff(s)) * x
                - sigma_s1 * phi_11 * model_s_combined)

        model_s1_source = self.model(x_s1, s1)
        guidance_s1 = self.compute_guidance_noise(x_s1, s1, self.guidance_network, ns) if self.guidance_network is not None else model_s1_source
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
        guidance_s = self.compute_guidance_noise(x, s, self.guidance_network, ns) if self.guidance_network is not None else model_s_source
        model_s_combined = model_s_source + guidance_s * self.scale1

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
        batch_size = x.size(0)  # divide x into batches
        # divide x into batches
        x = x.to('cpu')
        x_batches = x.split(batch_size)
        x_out_batches = []
        for x_batch in x_batches:
            device = x.device
            ts = torch.linspace(t_start, t_end, steps+1).to(device)
            # Initialize with a single first_order_update
            s = ts[0].unsqueeze(0)
            for step in range(1, steps+1):
                t = ts[step].unsqueeze(0)
                # s is the current noise step, t is the target noise step
                if step == 1:
                    x = self.first_order_update_guidance(x, s, t)
                else:
                    x = self.second_order_update_guidance(x, s, t)
                s = t
            x_out = x
            x_out_batches.append(x_out)
        return torch.cat(x_out_batches, dim=0)

if show_baseline:
# -----------------------------------------------------------------------
# 5. Vanilla Plot Train the Noise Predictor on the Target Domain
# -----------------------------------------------------------------------
    noise_predictor_target = NoisePredictor()
    train_diffusion_noise_prediction(noise_predictor_target,target_data_, vanilla_diffusion_epochs, batch_size, learning_rate,device)
    dpm_solver = DPM_Solver(noise_predictor_target, noise_schedule)
    x_init = torch.randn(10000, 2)
    x_out = dpm_solver.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
    plot_true_and_diffused(target_data_, x_out, target_labels_, None,
                           title_true_data="True Target Distribution (Trained on)", title_diffused_data="Vanilla DM Sampled Distribution")


# -----------------------------------------------------------------------
# 6. Baseline: Finetune the Noise Predictor on the Target Domain
# -----------------------------------------------------------------------
    # 1) Train the noise predictor on the source domain
    noise_predictor_target_finetune = NoisePredictor()
    train_diffusion_noise_prediction(noise_predictor_target_finetune, source_data, source_epochs, batch_size, learning_rate, device)
    dpm_solver = DPM_Solver(noise_predictor_target_finetune, noise_schedule)
    x_init = torch.randn(10000, 2)
    x_out = dpm_solver.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
    plot_true_and_diffused(true_data=source_data, diffused_data=x_out, true_labels=source_labels, diffused_labels=None
                           , title_true_data="True Source Distribution (Trained on)", title_diffused_data="Pre-trained DM Sampled Distribution")
    # 2) Finetune the noise predictor on the target domain
    train_diffusion_noise_prediction(noise_predictor_target_finetune, target_data_, finetune_diffusion_epochs, batch_size, learning_rate, device)
    dpm_solver = DPM_Solver(noise_predictor_target_finetune, noise_schedule)
    x_out = dpm_solver.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
    plot_true_and_diffused(true_data=target_data_, diffused_data=x_out, true_labels=target_labels_all_n[n_target_index], diffused_labels=None,
                           title_true_data="True Target Distribution (Fine-Tuned on)", title_diffused_data="Finetuned DM Sampled Distribution")



# -----------------------------------------------------------------------
# 7. Train the Domain Classifier and the Guidance Network
#   - Pseudo Code Algorithm 1 and Algorithm 3
# -----------------------------------------------------------------------
def train_domain_classifier(model, source_data, target_data, n_epochs=5, batch_size=512, lr=1e-4, device='cpu'):
    """
    Minimal training loop for domain classification on 2D data.
    """
    source_labels = torch.ones(source_data.size(0))
    target_labels = torch.zeros(target_data.size(0))

    dataset = TensorDataset(torch.cat([source_data, target_data]), torch.cat([source_labels, target_labels]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCELoss()

    model.train()
    model.to(device)

    for epoch in range(n_epochs):
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = model(x)
            loss = bce_loss(y_pred, y.unsqueeze(1).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0 and epoch > 0:
            print(f"[Classifier] epoch {epoch+1}/{n_epochs}, loss={loss.item():.4f}")
    model.to('cpu')
    return model


def train_guidance_network(
        guidance_network,
        classifier,
        noise_schedule,
        source_data,
        T=25,
        n_epochs=20,
        batch_size=512,
        lr=1e-4,
        device='cpu'
):
    """
    Trains 'guidance_network' using the objective:
        L(psi) = E_{x0, t, x_t} [|| h_psi(x_t, t) - c_omega(x0) ||^2_2],
    where x_t = alpha_t * x0 + sigma_t * eps.
    """
    dataset = TensorDataset(source_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    guidance_network.to(device)
    classifier.to(device)
    classifier.eval()  # Classifier is assumed pre-trained/frozen

    optimizer = optim.Adam(guidance_network.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # 2. Training loop
    count_update = 0
    for epoch in range(n_epochs):
        for batch in loader:
            x0 = batch[0].to(device)  # shape [batch, d]

            # Sample discrete t from {1,...,T} uniformly
            t_int = torch.randint(low=1, high=T + 1, size=(x0.size(0),), device=device)
            t_ = t_int.float() / float(T)

            # 3. Forward diffuse: x_t = alpha_t * x0 + sigma_t * eps
            alpha_t = noise_schedule.marginal_alpha(t_).unsqueeze(1)  # shape [batch, 1]
            sigma_t = noise_schedule.marginal_std(t_).unsqueeze(1)  # shape [batch, 1]
            eps = torch.randn_like(x0)
            x_t = alpha_t * x0 + sigma_t * eps

            # 4. Compute guidance_network outputs and domain classifier target
            with torch.no_grad():
                # c_omega(x0) is the classifier output for the original data
                classifier_out = classifier(x0)  # shape [batch, 1]

            guidance_out = guidance_network(torch.cat([x_t, t_.unsqueeze(1)], dim=1))

            # 5. Compute custom guidance loss (Algorithm 2)

            target = (1 - classifier_out) / classifier_out  # Algorithm 4 in the Paper
            # target = classifier_out # Algorithm 3 in the Paper
            # This is the original code, but I think it is wrong, the guidance loss should as in Algorithm 4
            loss = torch.mean(torch.sum((guidance_out - target) ** 2, dim=1))

            # 6. Update guidance_network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count_update += 1

        # (Optional) Print some progress
        # E.g., every 5 epochs
        if (epoch + 1) % 10 == 0:
            print(f"[Guidance] epoch {epoch + 1}/{n_epochs}, loss={loss.item():.4f}")

    # Return trained model weights
    #print(f"Total number of updates: {count_update}")
    guidance_network.to('cpu')
    return guidance_network


# -----------------------------------------------------------------------
# 8. Train the Domain Classifier and the Guidance Network with Regularization
#   - Pseudo Code Algorithm 2 and Algorithm 4
# -----------------------------------------------------------------------
def train_time_dependent_classifier(
        classifier_time_dependent,
        source_data,
        target_data,
        T=25,
        n_epochs=20,
        batch_size=512,
        lr=1e-4,
        device='cpu'):
    # Combine source and target data
    dataset = TensorDataset(
        torch.cat([source_data, target_data]),
        torch.cat([torch.ones(source_data.size(0)), torch.zeros(target_data.size(0))])
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

    classifier_time_dependent.to(device)
    optimizer = optim.Adam(classifier_time_dependent.parameters(), lr=lr)
    bce_loss = nn.BCELoss()

    for epoch in range(n_epochs):
        for x0, labels in loader:
            x0, labels = x0.to(device), labels.to(device)

            # Sample discrete time t and compute corresponding noise parameters
            t_int = torch.randint(low=1, high=T + 1, size=(x0.size(0),), device=device)
            t = t_int.float() / float(T)
            alpha_t = noise_schedule.marginal_alpha(t).unsqueeze(1)  # shape [batch, 1]
            sigma_t = noise_schedule.marginal_std(t).unsqueeze(1)  # shape [batch, 1]
            eps = torch.randn_like(x0)
            x_t = alpha_t * x0 + sigma_t * eps

            # Forward pass
            predictions = classifier_time_dependent(torch.cat([x_t, t_int.unsqueeze(1)], dim=1))
            loss = bce_loss(predictions, labels.unsqueeze(1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Time Dependent Classifier] epoch {epoch + 1}/{n_epochs}, loss={loss.item():.4f}")

    classifier_time_dependent.to('cpu')
    return classifier_time_dependent

def train_guidance_network_with_regularization(
        guidance_network,
        classifier,                   # c_ω(x0)
        classifier_time_dependent,   # c'_ω(x0, t)
        denoiser,                    # s_source(x_t, t)
        noise_schedule,
        source_data,
        target_data,
        eta1and2=[0.1, 0.1],         # [η1, η2]
        T=25,
        n_epochs=20,
        batch_size=512,
        lr=1e-4,
        device='cpu',
):
    """
    Trains the guidance network 'guidance_network' using:
      L_guidance + η1 * L_cycle + η2 * L_consistence,
    as described in the pseudocode (Algorithm 4).
    """

    source_batch_size = min(batch_size, source_data.size(0))
    target_batch_size = min(batch_size, target_data.size(0))

    # Move networks to device
    guidance_network.to(device)
    classifier.to(device)
    classifier_time_dependent.to(device)
    denoiser.to(device)

    # We assume classifier and denoiser are fixed (pre-trained)
    classifier.eval()
    classifier_time_dependent.eval()
    denoiser.eval()

    eta1 = eta1and2[0]
    use_loss_cycle = eta1 > 0.0

    eta2 = eta1and2[1]
    use_loss_consistence = eta2 > 0.0

    # Create data loaders
    source_loader = DataLoader(
        TensorDataset(source_data),
        batch_size=source_batch_size, shuffle=True, drop_last=True
    )
    target_loader = DataLoader(
        TensorDataset(target_data),
        batch_size=target_batch_size, shuffle=True, drop_last=True
    )
    # Have to adjust for the case when the source and target data have different sizes
    if len(source_loader) > len(target_loader):
        target_loader = cycle(target_loader)

    optimizer = optim.Adam(guidance_network.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    count_update = 0
    def grad_log_h(x_t, t_):
        """
        Computes ∇_{x_t} log h_ψ(x_t, t).
        Returns a tensor of the same shape as x_t.
        """
        x_t.requires_grad_(True)
        log_h_vals = torch.log(guidance_network(torch.cat([x_t, t_.unsqueeze(1)], dim=1)) + 1e-20).sum()
        grad_vals = torch.autograd.grad(log_h_vals, x_t, create_graph=False)[0]
        x_t.requires_grad_(False)
        return grad_vals



    for epoch in range(n_epochs):
        # Zip source_loader and target_loader to get matched mini‐batches
        for (source_batch,), (target_batch,) in zip(source_loader, target_loader):
            source_batch = source_batch.to(device)  # x0 from source
            target_batch = target_batch.to(device)  # x0' from target

            t_int_target = torch.randint(1, T + 1, (target_batch.size(0),), device=device)
            t_target = t_int_target.float() / T

            # Forward diffuse x0' using q(x_t'|x0')
            alpha_t_target = noise_schedule.marginal_alpha(t_target).unsqueeze(1)
            sigma_t_target = noise_schedule.marginal_std(t_target).unsqueeze(1)
            eps_target = torch.randn_like(target_batch)
            x_t_target = alpha_t_target * target_batch + sigma_t_target * eps_target


            # Sample time indices in [1..T]
            t_int_source = torch.randint(1, T + 1, (source_batch.size(0),), device=device)
            t_source = t_int_source.float() / T

            # Forward diffuse x0 using p(x_t|x0)
            alpha_t = noise_schedule.marginal_alpha(t_source).unsqueeze(1)
            sigma_t = noise_schedule.marginal_std(t_source).unsqueeze(1)
            eps = torch.randn_like(source_batch)
            x_t_source = alpha_t * source_batch + sigma_t * eps

            # ----------- 1) Guidance Loss (L_guidance) -----------
            # Compute classifier output c_ω(x0)
            with torch.no_grad():
                c_out = classifier(source_batch)  # shape [batch,1]
            # Target ratio = (1 - c_ω(x0)) / c_ω(x0)
            ratio = (1. - c_out) / (c_out )

            # Guidance output h_ψ(x_t, t)
            h_out = guidance_network(torch.cat([x_t_source, t_source.unsqueeze(1)], dim=1))
            # L_guidance = MSE( h_out, ratio )
            loss_guidance = mse_loss(h_out, ratio)

            # ----------- 2) Cycle Loss (L_cycle) -----------
            if use_loss_cycle:

                # c'_ω(x0', t)
                with torch.no_grad():
                    c_tdep_out = classifier_time_dependent(torch.cat([x_t_target, t_int_target.unsqueeze(1)], dim=1))
                    #c_tdep_out_paper = classifier_time_dependent(torch.cat([target_batch, t_int_target.unsqueeze(1)], dim=1))
                ratio_tdep = (1. - c_tdep_out) / (c_tdep_out + 1e-20)

                # h_ψ(x_t', t)
                h_out_target = guidance_network(torch.cat([x_t_target, t_target.unsqueeze(1)], dim=1))

                # L_cycle = MSE( h_out_target, c'_ω(x0', t) )
                loss_cycle = mse_loss(h_out_target, ratio_tdep)
            else:
                loss_cycle = torch.tensor(0., device=device)

            # ----------- 3) Consistency Loss (L_consistence) -----------
            if use_loss_consistence:
                # We reuse the same x_t_target from above (or create a new one if needed)
                # Make sure x_t_target, t_target, target_batch exist.
                # If we didn't sample them above, do it here:

                # 1) s_source(x'_t, t): the denoiser’s predicted noise
                with torch.no_grad():
                    s_source_out = denoiser(x_t_target, t_target)  # shape [batch, d]
                    sigma_t = noise_schedule.marginal_std(t_target).unsqueeze(1)  # shape [batch,1]
                    # ∇_x log q(x_t'| x0') = -(1 / σ_t)
                    score_source = - (1.0 / sigma_t) * s_source_out

                # 2) ∇ log h_ψ(x'_t, t)
                grad_log_h_val = grad_log_h(x_t_target, t_target)

                # 3) ∇ log q(x_t'| x0') (Gaussian assumption)
                #    => ∇_x' log q(...) = -(1 / σ_t^2) * (x'_t - α_t x0')
                grad_log_q = -(1.0 / (sigma_t_target**2)) * (x_t_target - alpha_t_target * target_batch)

                # L_consistence = MSE( s_source + grad_log_h_val - grad_log_q, 0 )
                # or equivalently MSE to the zero vector
                # We can treat it as the norm squared:
                consist_term = (score_source + grad_log_h_val - grad_log_q)
                loss_consistence = torch.mean(torch.sum(consist_term**2, dim=1))
            else:
                loss_consistence = torch.tensor(0., device=device)

            # 4) Total Loss
            total_loss = loss_guidance \
                         + eta1 * loss_cycle \
                         + eta2 * loss_consistence

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            count_update += 1

            # (Optional) Print progress
            if (epoch + 1) % 10 == 0:
                print(f"[Train Guidance with Rgul] epoch {epoch+1}/{n_epochs} "
                  f"L_guidance={loss_guidance.item():.4f} "
                  f"L_cycle={loss_cycle.item():.4f} "
                  f"L_consistence={loss_consistence.item():.4f}")

    # Move guidance network back to CPU and return
    guidance_network.to('cpu')
    #print(f"Total number of updates: {count_update}")
    return guidance_network

noise_predictor_tgdp = NoisePredictor()
train_diffusion_noise_prediction(noise_predictor_tgdp, source_data, source_epochs, batch_size, learning_rate, device)

classifier = Classifier()
classifier = train_domain_classifier(classifier, source_data, target_data_, guidance_epochs, batch_size, learning_rate, device)
evaluate_domain_classifier(model=classifier, source_data=source_data, target_data=target_data_, device=device)

if use_regularization :
    classifier_time_dependent = Classifier(input_dim=d+1)
    classifier_time_dependent = train_time_dependent_classifier(classifier_time_dependent, source_data, target_data_, T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device)
    guidance_network = GuidanceNetwork()
    guidance_network = train_guidance_network_with_regularization(guidance_network, classifier, classifier_time_dependent, noise_predictor_tgdp, noise_schedule, source_data, target_data_,
                                                                  eta1and2=eta1and2, T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size, lr=learning_rate, device=device)
else:
    guidance_network = GuidanceNetwork()
    guidance_network = train_guidance_network(guidance_network, classifier, noise_schedule, source_data,
                                              T=diffusion_steps, n_epochs=guidance_epochs, batch_size=batch_size,
                                              lr=learning_rate, device=device)

noise_predictor_tgdp.to('cpu')
# 1) Train the noise predictor on the source domain

dpm_solver = DPM_Solver(model_fn=noise_predictor_tgdp,guidance_scale=guidance_scale, noise_schedule=noise_schedule, guidance_network=guidance_network)
x_init = torch.randn(10000, 2)
x_out = dpm_solver.second_order_sample_guidance(x_init, steps=diffusion_steps, t_start=1.0, t_end=1e-3)
plot_true_and_diffused(true_data=target_data_, diffused_data=x_out, true_labels=target_labels_all_n[n_target_index], diffused_labels=None
                       , title_true_data="Target Distribution", title_diffused_data=f"Sampled Distribution from DM with Guidance")

pdf_vals = mixture_pdf(x_out, mu_T, sigma2)  # shape [N]
avg_likelihood = pdf_vals.mean().item()
print("Average likelihood:", avg_likelihood)

# -----------------------------------------------------------------------
# 9. Evaluate the Learned Density Ratio (Figure 2)
# -----------------------------------------------------------------------
if show_density_ratio:
    classifier.to('cpu')
    N = 1000
    t_rand = generate_samples(mu_T, sigma2, 10000)
    s_rand = generate_samples(mu_S, sigma2, 10000)
    x_rand = torch.cat([t_rand[0], s_rand[0]], dim=0)


    with torch.no_grad():
        # Oracle log-ratio
        ratio_oracle = true_ratio_oracle(x_rand, mu_S, mu_T, sigma2)
        log_ratio_oracle = torch.log(ratio_oracle + 1e-20)

        # Learned log-ratio
        cvals = classifier(x_rand)  # shape [N,1]
        ratio_learned = (1 - cvals) / (cvals + 1e-20)
        log_ratio_learned = torch.log(ratio_learned + 1e-20)
    plot_ration(x_rand, log_ratio_oracle, log_ratio_learned)


