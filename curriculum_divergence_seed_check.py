"""
Seed robustness check for the curriculum-divergence finding.

The original result (rho=-0.306, hardest-20% overlap=7.3%, well below the
20% expected by chance) came from ONE trained world model and ONE fixed
test set. Before presenting this as even a preliminary finding, we check
whether it holds across independently trained models.

"""

import numpy as np
from scipy.stats import spearmanr
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
ASSUMED_DAMPING_FOR_FIXED_CONTROLLER = 0.5

KD_GRID_LO, KD_GRID_HI, KD_GRID_N = 0.0, 15.0, 151   # matches original run (0.1 resolution)


def critical_kd(assumed_damping, g_over_l=G_OVER_L):
    return 2 * np.sqrt(g_over_l) - assumed_damping


def simulate_passive(theta0, omega0, damping, g_over_l=G_OVER_L, dt=DT, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    theta, omega = theta0, omega0
    for t in range(steps):
        states[t] = [theta, omega]
        alpha = -g_over_l * np.sin(theta) - damping * omega
        omega += alpha * dt
        theta += omega * dt
    return states


def simulate_controlled(theta0, omega0, true_damping, kd, g_over_l=G_OVER_L, dt=DT, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    theta, omega = theta0, omega0
    for t in range(steps):
        states[t] = [theta, omega]
        u = -kd * omega
        alpha = -g_over_l * np.sin(theta) - true_damping * omega + u
        omega += alpha * dt
        theta += omega * dt
    return states


def best_kd_for_trajectory(theta0, omega0, true_damping, kd_grid, horizon,
                            g_over_l=G_OVER_L, dt=DT):
    best_cost = np.inf
    best_kd = kd_grid[0]
    for kd in kd_grid:
        theta, omega = theta0, omega0
        cost_sum = 0.0
        for t in range(horizon):
            u = -kd * omega
            alpha = -g_over_l * np.sin(theta) - true_damping * omega + u
            omega += alpha * dt
            theta += omega * dt
            cost_sum += theta ** 2 + 0.1 * omega ** 2
        cost = cost_sum / horizon
        if cost < best_cost:
            best_cost = cost
            best_kd = kd
    return best_kd, best_cost


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


KD_SEARCH_GRID_BASE = np.linspace(KD_GRID_LO, KD_GRID_HI, KD_GRID_N)

per_seed_results = []

for seed in SEEDS:
    print(f"\n{'=' * 70}")
    print(f"SEED = {seed}")
    print(f"{'=' * 70}")
    rng_base = seed * 100

    damping_train = np.full(NUM_TRAIN, 0.5)
    init_theta_train = np.random.RandomState(rng_base + 1).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TRAIN)
    init_omega_train = np.random.RandomState(rng_base + 4).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TRAIN)
    states_train = np.array([simulate_passive(th0, om0, 0.5)
                              for th0, om0 in zip(init_theta_train, init_omega_train)])

    damping_test = np.random.RandomState(rng_base + 2).uniform(0.1, 0.9, NUM_TEST)
    init_theta_test = np.random.RandomState(rng_base + 3).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TEST)
    init_omega_test = np.random.RandomState(rng_base + 5).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TEST)
    states_test_passive = np.array([simulate_passive(th0, om0, d)
                                     for th0, om0, d in zip(init_theta_test, init_omega_test, damping_test)])

    X_train = states_train[:, :-1, :].reshape(-1, 2)
    y_train = states_train[:, 1:, :].reshape(-1, 2)

    print("Training world model...")
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

    rollout_error = rollout_relative_error(states_test_passive, HORIZON, lambda s: nn_forward(s, *W_trained))

    KD_FIXED = critical_kd(ASSUMED_DAMPING_FOR_FIXED_CONTROLLER)
    kd_grid = KD_SEARCH_GRID_BASE.copy()
    if not np.any(np.isclose(kd_grid, KD_FIXED, atol=(KD_GRID_HI - KD_GRID_LO) / (KD_GRID_N - 1) / 2)):
        kd_grid = np.sort(np.append(kd_grid, KD_FIXED))

    print("Computing fixed-controller and grid-searched optimal-controller costs...")
    states_test_fixed_ctrl = np.array([simulate_controlled(th0, om0, d, KD_FIXED)
                                        for th0, om0, d in zip(init_theta_test, init_omega_test, damping_test)])

    def controller_cost(states, horizon):
        theta_seq = states[:, 1:horizon + 1, 0]
        omega_seq = states[:, 1:horizon + 1, 1]
        return np.mean(theta_seq ** 2 + 0.1 * omega_seq ** 2, axis=1)

    cost_fixed = controller_cost(states_test_fixed_ctrl, HORIZON)

    cost_optimal = np.zeros(NUM_TEST)
    for i, (th0, om0, d) in enumerate(zip(init_theta_test, init_omega_test, damping_test)):
        _, best_cost = best_kd_for_trajectory(th0, om0, d, kd_grid, HORIZON)
        cost_optimal[i] = best_cost

    regret = cost_fixed - cost_optimal
    frac_negative = np.mean(regret < -1e-9)

    rho, pval = spearmanr(rollout_error, regret)
    k = int(0.2 * NUM_TEST)
    hardest_by_error = set(np.argsort(-rollout_error)[:k])
    hardest_by_regret = set(np.argsort(-regret)[:k])
    overlap_frac = len(hardest_by_error & hardest_by_regret) / k

    print(f"  frac_negative regret (sanity, should be ~0): {frac_negative:.2%}")
    print(f"  Spearman(rollout_error, regret): rho={rho:.4f} (p={pval:.2e})")
    print(f"  Hardest-20% overlap: {overlap_frac:.1%} (chance level: 20.0%)")

    per_seed_results.append({'seed': seed, 'rho': rho, 'pval': pval, 'overlap': overlap_frac,
                              'frac_negative': frac_negative})

print(f"\n{'=' * 70}")
print(f"AGGREGATE ACROSS {len(SEEDS)} SEEDS")
print(f"{'=' * 70}")
print(f"{'seed':>6} {'rho':>10} {'p-value':>12} {'overlap%':>10}")
for r in per_seed_results:
    print(f"{r['seed']:>6} {r['rho']:>10.4f} {r['pval']:>12.2e} {r['overlap']*100:>9.1f}%")

rhos = np.array([r['rho'] for r in per_seed_results])
overlaps = np.array([r['overlap'] for r in per_seed_results])
print(f"\nrho:     mean={rhos.mean():.4f}  std={rhos.std():.4f}  "
      f"all negative: {np.all(rhos < 0)}")
print(f"overlap: mean={overlaps.mean():.1%}  std={overlaps.std():.1%}  "
      f"all below 20% chance level: {np.all(overlaps < 0.20)}")
