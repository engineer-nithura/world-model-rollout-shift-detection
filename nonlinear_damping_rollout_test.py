"""
Same closed-loop rollout U-shape test, but on a NONLINEAR system.

Original system: linear damping force = -damping * v
  -> the ODE is linear, damping is identifiable in closed form from a
     short window of true states (this is why your very first Raw/Latent
     baselines were suspiciously strong before you even trained anything).

New system: quadratic (velocity-squared) damping force = -damping * v * |v|
  -> this is a standard "nonlinear drag" model (like air resistance at
     higher speeds). No closed-form solution exists for x(t); damping
     can't be trivially read off a short window anymore.

Why this test matters: if the closed-loop rollout U-shape (error rises
with distance from the training regime, and requires training) shows up
here too, that's evidence the effect is a general property of "trained
model meets unfamiliar dynamics" -- not an artifact specific to your
original linear/closed-form-solvable system. If it breaks down here,
that's ALSO informative: it would mean the effect depends on the test
regime being "close" to the training regime in some specific way that
doesn't hold once the dynamics are more complex.

Kept at a single bottleneck size (latent_dim=4) and single horizon (20)
to isolate the one new variable (system nonlinearity) rather than
re-sweeping everything at once.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor

SEED = 0
np.random.seed(SEED)

# Config
LATENT_DIM = 4
HORIZON = 20
NUM_TRAIN = 3000
NUM_TEST = 1500
HIDDEN_DIM = 128
TRAJ_STEPS = 40
CLIP_MAG = 1e3
N_BINS = 9
EPS = 1e-3


# Physics: NONLINEAR damping 
def simulate_trajectory_nonlinear(x0, v0, damping, k=1.0, mass=1.0, dt=0.1, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    x, v = x0, v0
    for t in range(steps):
        states[t] = [x, v]
        # Quadratic drag: force opposes velocity, scales with v^2 (via v*|v| to keep sign correct)
        a = (-k * x - damping * v * np.abs(v)) / mass
        v += a * dt
        x += v * dt
    return states


print("Generating data (NONLINEAR quadratic-damping system)...")
damping_train = np.full(NUM_TRAIN, 0.5)
init_train = np.random.uniform(-1, 1, (NUM_TRAIN, 2))
states_train = np.array([simulate_trajectory_nonlinear(x0, v0, 0.5) for x0, v0 in init_train])

damping_test = np.random.uniform(0.1, 0.9, NUM_TEST)
init_test = np.random.uniform(-1, 1, (NUM_TEST, 2))
states_test = np.array([simulate_trajectory_nonlinear(x0, v0, d) for (x0, v0), d in zip(init_test, damping_test)])

X_train = states_train[:, :-1, :].reshape(-1, 2)
y_train = states_train[:, 1:, :].reshape(-1, 2)

# Sanity check: confirm trajectories are well-behaved (no NaN/blowup in the ground truth itself)
assert np.all(np.isfinite(states_train)) and np.all(np.isfinite(states_test)), \
    "Ground-truth trajectories contain NaN/Inf -- reduce dt or damping range."
print(f"  train state magnitude range: [{np.abs(states_train).min():.3f}, {np.abs(states_train).max():.3f}]")
print(f"  test  state magnitude range: [{np.abs(states_test).min():.3f}, {np.abs(states_test).max():.3f}]")


#  Model
def forward(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    z = np.tanh(h1 @ W2 + b2)
    pred = z @ W3 + b3
    return z, pred

print(f"\nTraining world model (latent_dim={LATENT_DIM}) on nonlinear system...")
model = MLPRegressor(
    hidden_layer_sizes=(HIDDEN_DIM, LATENT_DIM),
    activation='tanh', solver='adam', alpha=0.0,
    learning_rate_init=0.001, max_iter=200, batch_size=256,
    random_state=42, shuffle=True
)
model.fit(X_train, y_train)
W_trained = (model.coefs_[0], model.intercepts_[0],
             model.coefs_[1], model.intercepts_[1],
             model.coefs_[2], model.intercepts_[2])

rng = np.random.RandomState(123)
W1_rand = rng.uniform(-1, 1, (2, HIDDEN_DIM)) / np.sqrt(2)
b1_rand = np.zeros(HIDDEN_DIM)
W2_rand = rng.uniform(-1, 1, (HIDDEN_DIM, LATENT_DIM)) / np.sqrt(HIDDEN_DIM)
b2_rand = np.zeros(LATENT_DIM)
W3_rand = rng.uniform(-1, 1, (LATENT_DIM, 2)) / np.sqrt(LATENT_DIM)
b3_rand = np.zeros(2)
W_untrained = (W1_rand, b1_rand, W2_rand, b2_rand, W3_rand, b3_rand)

WEIGHT_SETS = {'trained': W_trained, 'untrained': W_untrained}


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

print(f"\n{'=' * 70}")
print(f"Nonlinear (quadratic) damping -- closed-loop relative error, horizon={HORIZON}")
print(f"{'=' * 70}")

fit_results = {}
for weight_name, weights in WEIGHT_SETS.items():
    errs = mean_relative_rollout_error_per_trajectory(HORIZON, weights)
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
    print(f"  -> quadratic fit: a(curvature)={a:>8.4f}  b(asymmetry)={b_coef:>8.4f}  c(baseline@0.5)={c:>8.4f}")
    min_bin = np.nanargmin(bin_means)
    print(f"  -> min-error bin: damping={bin_centers[min_bin]:.3f} "
          f"({'near 0.5' if abs(bin_centers[min_bin] - 0.5) < 0.15 else 'NOT near 0.5'})")
    fit_results[weight_name] = (a, b_coef, c)

print(f"\n{'=' * 70}")
print("Comparison to the ORIGINAL linear-damping system (dim=4, horizon=20):")
print("  linear system had: a_trained=1.1391, a_untrained=-0.0028")
print(f"  this (nonlinear) system: a_trained={fit_results['trained'][0]:.4f}, "
      f"a_untrained={fit_results['untrained'][0]:.4f}")
print(f"{'=' * 70}")
