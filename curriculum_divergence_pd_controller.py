"""
Previous divergence result used a single-gain controller family
(u = -Kd*omega, damping augmentation only). This tests a genuinely
different controller CLASS: full PD control, u = -Kp*theta - Kd*omega,
with two independently tunable gains that can shape both the restoring
force and the damping.

"""

import numpy as np
from scipy.stats import spearmanr
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
DT = 0.05
G_OVER_L = 9.8
CLIP_MAG = 1e3
N_BINS = 9
EPS = 1e-3
THETA0_RANGE = 2.2
OMEGA0_RANGE = 1.0
ASSUMED_DAMPING = 0.5

# Fixed PD controller: Kp is an arbitrary but reasonable moderate stiffness
# augmentation; Kd is chosen for critical damping of the AUGMENTED system,
# assuming damping=0.5 (same information the world model has).
KP_FIXED = 2.0
KD_FIXED = 2 * np.sqrt(G_OVER_L + KP_FIXED) - ASSUMED_DAMPING

# 2D grid search over (Kp, Kd). Coarser per-dimension than the 1D search
# (144 total combinations vs 151) to keep 2D search cost comparable to the
# earlier 1D run at the same NUM_TEST.
KP_GRID = np.arange(0.0, 9.0, 1.0)     # 9 values: 0..8
KD_GRID = np.arange(0.0, 16.0, 1.0)    # 16 values: 0..15


def simulate_passive(theta0, omega0, damping, g_over_l=G_OVER_L, dt=DT, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    theta, omega = theta0, omega0
    for t in range(steps):
        states[t] = [theta, omega]
        alpha = -g_over_l * np.sin(theta) - damping * omega
        omega += alpha * dt
        theta += omega * dt
    return states


def simulate_pd_controlled(theta0, omega0, true_damping, kp, kd,
                            g_over_l=G_OVER_L, dt=DT, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    theta, omega = theta0, omega0
    for t in range(steps):
        states[t] = [theta, omega]
        u = -kp * theta - kd * omega
        alpha = -g_over_l * np.sin(theta) - true_damping * omega + u
        omega += alpha * dt
        theta += omega * dt
    return states


def controller_cost_from_states(states, horizon):
    theta_seq = states[:, 1:horizon + 1, 0]
    omega_seq = states[:, 1:horizon + 1, 1]
    return np.mean(theta_seq ** 2 + 0.1 * omega_seq ** 2, axis=1)


def best_pd_for_trajectory(theta0, omega0, true_damping, kp_grid, kd_grid, horizon,
                            fixed_pair=None, g_over_l=G_OVER_L, dt=DT):
    """Exhaustive search over the (Kp, Kd) grid, plus the fixed controller's
    exact pair explicitly included (fixed_pair), guaranteeing this can never
    find something worse than the fixed controller: regret >= 0 by
    construction, same guarantee as the 1D search."""
    candidates = [(kp, kd) for kp in kp_grid for kd in kd_grid]
    if fixed_pair is not None and fixed_pair not in candidates:
        candidates.append(fixed_pair)

    best_cost = np.inf
    best_pair = candidates[0]
    for kp, kd in candidates:
        theta, omega = theta0, omega0
        cost_sum = 0.0
        for t in range(horizon):
            u = -kp * theta - kd * omega
            alpha = -g_over_l * np.sin(theta) - true_damping * omega + u
            omega += alpha * dt
            theta += omega * dt
            cost_sum += theta ** 2 + 0.1 * omega ** 2
        cost = cost_sum / horizon
        if cost < best_cost:
            best_cost = cost
            best_pair = (kp, kd)
    return best_pair, best_cost


def nn_forward(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    z = np.tanh(h1 @ W2 + b2)
    return z @ W3 + b3


def rollout_relative_error(states, horizon, predict_fn):
    mean_rel_err = np.zeros(states.shape[0])
    for i in range(states.shape[0]):
        true_points = states[i, 0:horizon + 1]
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


print(f"Fixed PD controller: Kp={KP_FIXED:.3f}, Kd={KD_FIXED:.3f} (assumes damping={ASSUMED_DAMPING})")
print(f"Search grid: {len(KP_GRID)} x {len(KD_GRID)} = {len(KP_GRID) * len(KD_GRID)} combinations "
      f"(+ fixed pair explicitly included)")
print("\nGenerating data...")

damping_train = np.full(NUM_TRAIN, 0.5)
init_theta_train = np.random.RandomState(1).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TRAIN)
init_omega_train = np.random.RandomState(4).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TRAIN)
states_train = np.array([simulate_passive(th0, om0, 0.5)
                          for th0, om0 in zip(init_theta_train, init_omega_train)])

damping_test = np.random.RandomState(2).uniform(0.1, 0.9, NUM_TEST)
init_theta_test = np.random.RandomState(3).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TEST)
init_omega_test = np.random.RandomState(5).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TEST)
states_test_passive = np.array([simulate_passive(th0, om0, d)
                                 for th0, om0, d in zip(init_theta_test, init_omega_test, damping_test)])

X_train = states_train[:, :-1, :].reshape(-1, 2)
y_train = states_train[:, 1:, :].reshape(-1, 2)

print(f"Training world model (latent_dim={LATENT_DIM})...")
model = MLPRegressor(
    hidden_layer_sizes=(HIDDEN_DIM, LATENT_DIM),
    activation='tanh', solver='adam', alpha=0.0,
    learning_rate_init=0.001, max_iter=300, batch_size=256,
    random_state=42, shuffle=True
)
model.fit(X_train, y_train)
W_trained = (model.coefs_[0], model.intercepts_[0],
             model.coefs_[1], model.intercepts_[1],
             model.coefs_[2], model.intercepts_[2])

print("Computing world-model rollout error (passive trajectories)...")
rollout_error = rollout_relative_error(states_test_passive, HORIZON, lambda s: nn_forward(s, *W_trained))

print("Computing fixed-PD-controller costs...")
states_test_fixed_ctrl = np.array([simulate_pd_controlled(th0, om0, d, KP_FIXED, KD_FIXED)
                                    for th0, om0, d in zip(init_theta_test, init_omega_test, damping_test)])
cost_fixed = controller_cost_from_states(states_test_fixed_ctrl, HORIZON)

print("Grid-searching optimal (Kp, Kd) per trajectory (this is the slow step)...")
cost_optimal = np.zeros(NUM_TEST)
for i, (th0, om0, d) in enumerate(zip(init_theta_test, init_omega_test, damping_test)):
    _, best_cost = best_pd_for_trajectory(th0, om0, d, KP_GRID, KD_GRID, HORIZON,
                                           fixed_pair=(KP_FIXED, KD_FIXED))
    cost_optimal[i] = best_cost
    if (i + 1) % 300 == 0:
        print(f"  ...{i + 1}/{NUM_TEST} trajectories optimized")

regret = cost_fixed - cost_optimal
frac_negative = np.mean(regret < -1e-9)
print(f"\nSanity check: fraction of trajectories with negative regret: {frac_negative:.2%} "
      f"(should be ~0, guaranteed by construction)")

rho, pval = spearmanr(rollout_error, regret)
k = int(0.2 * NUM_TEST)
hardest_by_error = set(np.argsort(-rollout_error)[:k])
hardest_by_regret = set(np.argsort(-regret)[:k])
overlap_frac = len(hardest_by_error & hardest_by_regret) / k

print(f"\n{'=' * 70}")
print("RESULT: rollout error vs. PD-controller regret")
print(f"{'=' * 70}")
print(f"Spearman(rollout_error, regret): rho={rho:.4f}  (p={pval:.2e})")
print(f"Hardest-20% overlap: {overlap_frac:.1%}  (expected by chance: 20.0%)")

print(f"\n{'=' * 70}")
print("COMPARISON TO SINGLE-GAIN CONTROLLER RESULT (from prior run)")
print(f"{'=' * 70}")
print("  single-gain (u=-Kd*omega):  rho=-0.3061, overlap=7.3%")
print(f"  PD (u=-Kp*theta-Kd*omega): rho={rho:.4f}, overlap={overlap_frac:.1%}")
