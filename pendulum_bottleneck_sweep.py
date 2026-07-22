"""
Bottleneck-capacity sweep, on the LARGE-ANGLE PENDULUM system . Same methodology as
bottleneck_sweep_rollout_error.py, same rigor (untrained control,
relative error, quadratic fit).

"""

import numpy as np
from sklearn.neural_network import MLPRegressor

SEED = 0
np.random.seed(SEED)

# Config
LATENT_DIMS = [2, 4, 8, 16]
HORIZON = 20
NUM_TRAIN = 3000
NUM_TEST = 1500
HIDDEN_DIM = 128
TRAJ_STEPS = 40
DT = 0.05
G_OVER_L = 9.8
CLIP_MAG = 1e3
N_BINS = 9
EPS = 1e-3
THETA0_RANGE = 2.2
OMEGA0_RANGE = 1.0


# Physics logic for large angle pendulum
def simulate_pendulum(theta0, omega0, damping, g_over_l=G_OVER_L, dt=DT, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    theta, omega = theta0, omega0
    for t in range(steps):
        states[t] = [theta, omega]
        alpha = -g_over_l * np.sin(theta) - damping * omega
        omega += alpha * dt
        theta += omega * dt
    return states


print("Generating data (shared across all bottleneck sizes, large-angle pendulum)...")
damping_train = np.full(NUM_TRAIN, 0.5)
init_theta_train = np.random.RandomState(1).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TRAIN)
init_omega_train = np.random.RandomState(4).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TRAIN)
states_train = np.array([simulate_pendulum(th0, om0, 0.5)
                          for th0, om0 in zip(init_theta_train, init_omega_train)])

damping_test = np.random.RandomState(2).uniform(0.1, 0.9, NUM_TEST)
init_theta_test = np.random.RandomState(3).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TEST)
init_omega_test = np.random.RandomState(5).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TEST)
states_test = np.array([simulate_pendulum(th0, om0, d)
                         for th0, om0, d in zip(init_theta_test, init_omega_test, damping_test)])

X_train = states_train[:, :-1, :].reshape(-1, 2)
y_train = states_train[:, 1:, :].reshape(-1, 2)


#  Model
def forward(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    z = np.tanh(h1 @ W2 + b2)
    pred = z @ W3 + b3
    return z, pred


def train_model(latent_dim):
    model = MLPRegressor(
        hidden_layer_sizes=(HIDDEN_DIM, latent_dim),
        activation='tanh', solver='adam', alpha=0.0,
        learning_rate_init=0.001, max_iter=300, batch_size=256,
        random_state=42, shuffle=True
    )
    model.fit(X_train, y_train)
    return (model.coefs_[0], model.intercepts_[0],
             model.coefs_[1], model.intercepts_[1],
             model.coefs_[2], model.intercepts_[2])


def random_weights(latent_dim, seed=123):
    rng = np.random.RandomState(seed)
    W1 = rng.uniform(-1, 1, (2, HIDDEN_DIM)) / np.sqrt(2)
    b1 = np.zeros(HIDDEN_DIM)
    W2 = rng.uniform(-1, 1, (HIDDEN_DIM, latent_dim)) / np.sqrt(HIDDEN_DIM)
    b2 = np.zeros(latent_dim)
    W3 = rng.uniform(-1, 1, (latent_dim, 2)) / np.sqrt(latent_dim)
    b3 = np.zeros(2)
    return (W1, b1, W2, b2, W3, b3)


# Closed-loop mean relative rollout error per trajectory
def mean_relative_rollout_error_per_trajectory(horizon, weights):
    W1, b1, W2, b2, W3, b3 = weights
    mean_rel_err = np.zeros(states_test.shape[0])

    for i in range(states_test.shape[0]):
        true_points = states_test[i, 0:horizon + 1]
        current_state = true_points[0]
        step_errs = []
        diverged = False
        for step in range(horizon):
            true_next = true_points[step + 1]
            if not diverged:
                _, pred_next = forward(current_state, W1, b1, W2, b2, W3, b3)
                if not np.all(np.isfinite(pred_next)) or np.max(np.abs(pred_next)) > CLIP_MAG:
                    pred_next = np.clip(np.nan_to_num(pred_next, nan=CLIP_MAG, posinf=CLIP_MAG, neginf=-CLIP_MAG),
                                         -CLIP_MAG, CLIP_MAG)
                    diverged = True
            else:
                pred_next = current_state
            abs_err = np.linalg.norm(pred_next - true_next)
            rel_err = abs_err / (np.linalg.norm(true_next) + EPS)
            step_errs.append(rel_err)
            current_state = pred_next
        mean_rel_err[i] = np.mean(step_errs)

    return mean_rel_err


# Binning + sharpness summary 
bin_edges = np.linspace(0.1, 0.9, N_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_idx = np.digitize(damping_test, bin_edges) - 1
bin_idx = np.clip(bin_idx, 0, N_BINS - 1)


def summarize(errs, weight_name):
    print(f"\n  [{weight_name}]")
    print(f"  {'damping bin':>14} {'n':>6} {'mean rel err':>14}")
    bin_means = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = bin_idx == b
        n = mask.sum()
        if n == 0:
            continue
        bin_means[b] = errs[mask].mean()
        print(f"  {bin_centers[b]:>14.3f} {n:>6} {bin_means[b]:>14.4f}")

    valid = ~np.isnan(bin_means)
    x = bin_centers[valid] - 0.5
    y = bin_means[valid]
    a, b_coef, c = np.polyfit(x, y, 2)
    print(f"  -> quadratic fit: a(curvature)={a:>7.4f}  b(asymmetry)={b_coef:>7.4f}  c(baseline@0.5)={c:>7.4f}")
    return a, b_coef, c, bin_means.tolist()


results_summary = {}

for latent_dim in LATENT_DIMS:
    print(f"\n{'=' * 75}")
    print(f"Bottleneck dim = {latent_dim}  (horizon={HORIZON}, large-angle pendulum)")
    print(f"{'=' * 75}")

    print(f"Training world model (dim={latent_dim})...")
    W_trained = train_model(latent_dim)
    W_untrained = random_weights(latent_dim)

    errs_trained = mean_relative_rollout_error_per_trajectory(HORIZON, W_trained)
    a_t, b_t, c_t, bins_t = summarize(errs_trained, 'trained')

    errs_untrained = mean_relative_rollout_error_per_trajectory(HORIZON, W_untrained)
    a_u, b_u, c_u, bins_u = summarize(errs_untrained, 'untrained')

    results_summary[latent_dim] = {
        'trained': {'a': a_t, 'b': b_t, 'c': c_t, 'bin_means': bins_t},
        'untrained': {'a': a_u, 'b': b_u, 'c': c_u, 'bin_means': bins_u},
    }

print(f"\n{'=' * 75}")
print("SUMMARY: curvature (a) by bottleneck dimension (large-angle pendulum)")
print(f"{'=' * 75}")
print(f"{'dim':>6} {'a_trained':>12} {'a_untrained':>14} {'b_trained (asymmetry)':>24}")
for d in LATENT_DIMS:
    a_t = results_summary[d]['trained']['a']
    b_t = results_summary[d]['trained']['b']
    a_u = results_summary[d]['untrained']['a']
    print(f"{d:>6} {a_t:>12.4f} {a_u:>14.4f} {b_t:>24.4f}")
