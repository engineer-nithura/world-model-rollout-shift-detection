"""
Seed robustness check for the large-angle pendulum result.

Reruns data generation + NN training + linear observer fitting + the
closed-loop relative-error-by-bin test across multiple seeds, and
reports a (curvature) and c (baseline error at damping=0.5) for all
three models (NN trained, NN untrained, linear observer) at each seed,
plus mean +/- std across seeds.

"""

import numpy as np
from sklearn.neural_network import MLPRegressor

# Config 
SEEDS = [0, 1, 2, 3, 4]
LATENT_DIM = 4
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


def simulate_pendulum(theta0, omega0, damping, g_over_l=G_OVER_L, dt=DT, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    theta, omega = theta0, omega0
    for t in range(steps):
        states[t] = [theta, omega]
        alpha = -g_over_l * np.sin(theta) - damping * omega
        omega += alpha * dt
        theta += omega * dt
    return states


def nn_forward(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    z = np.tanh(h1 @ W2 + b2)
    return z @ W3 + b3


def fit_linear_observer(X_train, y_train):
    A_T, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return A_T


def rollout_relative_error(states_test, damping_test, horizon, predict_fn):
    mean_rel_err = np.zeros(states_test.shape[0])
    for i in range(states_test.shape[0]):
        true_points = states_test[i, 0:horizon + 1]
        current_state = true_points[0]
        step_errs = []
        diverged = False
        for step in range(horizon):
            true_next = true_points[step + 1]
            if not diverged:
                pred_next = predict_fn(current_state)
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


def fit_quadratic(errs, damping_test):
    bin_edges = np.linspace(0.1, 0.9, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(damping_test, bin_edges) - 1, 0, N_BINS - 1)
    bin_means = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_means[b] = errs[mask].mean()
    valid = ~np.isnan(bin_means)
    x = bin_centers[valid] - 0.5
    y = bin_means[valid]
    a, b_coef, c = np.polyfit(x, y, 2)
    return a, b_coef, c


results = {'nn_trained': [], 'nn_untrained': [], 'linear_observer': []}

for seed in SEEDS:
    print(f"\n{'=' * 70}")
    print(f"SEED = {seed}")
    print(f"{'=' * 70}")
    rng_base = seed * 100  # offset RandomStates per seed so streams don't overlap

    damping_train = np.full(NUM_TRAIN, 0.5)
    init_theta_train = np.random.RandomState(rng_base + 1).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TRAIN)
    init_omega_train = np.random.RandomState(rng_base + 4).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TRAIN)
    states_train = np.array([simulate_pendulum(th0, om0, 0.5)
                              for th0, om0 in zip(init_theta_train, init_omega_train)])

    damping_test = np.random.RandomState(rng_base + 2).uniform(0.1, 0.9, NUM_TEST)
    init_theta_test = np.random.RandomState(rng_base + 3).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TEST)
    init_omega_test = np.random.RandomState(rng_base + 5).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TEST)
    states_test = np.array([simulate_pendulum(th0, om0, d)
                             for th0, om0, d in zip(init_theta_test, init_omega_test, damping_test)])

    X_train = states_train[:, :-1, :].reshape(-1, 2)
    y_train = states_train[:, 1:, :].reshape(-1, 2)

    print("Training neural world model...")
    model = MLPRegressor(
        hidden_layer_sizes=(HIDDEN_DIM, LATENT_DIM),
        activation='tanh', solver='adam', alpha=0.0,
        learning_rate_init=0.001, max_iter=300, batch_size=256,
        random_state=seed, shuffle=True
    )
    model.fit(X_train, y_train)
    W_trained = (model.coefs_[0], model.intercepts_[0],
                 model.coefs_[1], model.intercepts_[1],
                 model.coefs_[2], model.intercepts_[2])

    rng = np.random.RandomState(rng_base + 123)
    W1_rand = rng.uniform(-1, 1, (2, HIDDEN_DIM)) / np.sqrt(2)
    b1_rand = np.zeros(HIDDEN_DIM)
    W2_rand = rng.uniform(-1, 1, (HIDDEN_DIM, LATENT_DIM)) / np.sqrt(HIDDEN_DIM)
    b2_rand = np.zeros(LATENT_DIM)
    W3_rand = rng.uniform(-1, 1, (LATENT_DIM, 2)) / np.sqrt(LATENT_DIM)
    b3_rand = np.zeros(2)
    W_untrained = (W1_rand, b1_rand, W2_rand, b2_rand, W3_rand, b3_rand)

    A_T = fit_linear_observer(X_train, y_train)

    errs_nn_t = rollout_relative_error(states_test, damping_test, HORIZON, lambda s: nn_forward(s, *W_trained))
    a_nn_t, b_nn_t, c_nn_t = fit_quadratic(errs_nn_t, damping_test)

    errs_nn_u = rollout_relative_error(states_test, damping_test, HORIZON, lambda s: nn_forward(s, *W_untrained))
    a_nn_u, b_nn_u, c_nn_u = fit_quadratic(errs_nn_u, damping_test)

    errs_lo = rollout_relative_error(states_test, damping_test, HORIZON, lambda s: s @ A_T)
    a_lo, b_lo, c_lo = fit_quadratic(errs_lo, damping_test)

    print(f"  NN trained:       a={a_nn_t:.4f}  c={c_nn_t:.4f}")
    print(f"  NN untrained:     a={a_nn_u:.4f}  c={c_nn_u:.4f}")
    print(f"  Linear observer:  a={a_lo:.4f}  c={c_lo:.4f}")

    results['nn_trained'].append((a_nn_t, c_nn_t))
    results['nn_untrained'].append((a_nn_u, c_nn_u))
    results['linear_observer'].append((a_lo, c_lo))

# Aggregate
print(f"\n{'=' * 70}")
print(f"AGGREGATE ACROSS {len(SEEDS)} SEEDS")
print(f"{'=' * 70}")
print(f"{'model':>18} {'a mean':>10} {'a std':>10} {'c mean':>10} {'c std':>10}")
for name, vals in results.items():
    arr = np.array(vals)
    a_mean, a_std = arr[:, 0].mean(), arr[:, 0].std()
    c_mean, c_std = arr[:, 1].mean(), arr[:, 1].std()
    print(f"{name:>18} {a_mean:>10.4f} {a_std:>10.4f} {c_mean:>10.4f} {c_std:>10.4f}")
