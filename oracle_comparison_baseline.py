import numpy as np

#  Config (matches pendulum_seed_robustness_check.py)
SEEDS = [0, 1, 2, 3, 4]
HORIZON = 20
NUM_TEST = 1500
TRAJ_STEPS = 40
DT = 0.05
G_OVER_L = 9.8
CLIP_MAG = 1e3
N_BINS = 9
EPS = 1e-3
THETA0_RANGE = 2.2
OMEGA0_RANGE = 1.0
ASSUMED_DAMPING = 0.5  

# NN and linear-observer results from the actual 5-seed run, for direct comparison
NN_TRAINED = {  # (a, c) per seed
    0: (0.3106, 0.0999), 1: (0.3512, 0.0563), 2: (0.2828, 0.0951),
    3: (0.3816, 0.1262), 4: (0.3508, 0.0991),
}
LINEAR_OBSERVER = {
    0: (0.1857, 0.1848), 1: (0.1655, 0.1834), 2: (0.2651, 0.1761),
    3: (0.2005, 0.1817), 4: (0.1881, 0.1852),
}


def simulate_pendulum(theta0, omega0, damping, g_over_l=G_OVER_L, dt=DT, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    theta, omega = theta0, omega0
    for t in range(steps):
        states[t] = [theta, omega]
        alpha = -g_over_l * np.sin(theta) - damping * omega
        omega += alpha * dt
        theta += omega * dt
    return states


def oracle_step(state, assumed_damping=ASSUMED_DAMPING, g_over_l=G_OVER_L, dt=DT):
    theta, omega = state
    alpha = -g_over_l * np.sin(theta) - assumed_damping * omega
    omega_next = omega + alpha * dt
    theta_next = theta + omega_next * dt
    return np.array([theta_next, omega_next])


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


oracle_results = {}

for seed in SEEDS:
    rng_base = seed * 100
    damping_test = np.random.RandomState(rng_base + 2).uniform(0.1, 0.9, NUM_TEST)
    init_theta_test = np.random.RandomState(rng_base + 3).uniform(-THETA0_RANGE, THETA0_RANGE, NUM_TEST)
    init_omega_test = np.random.RandomState(rng_base + 5).uniform(-OMEGA0_RANGE, OMEGA0_RANGE, NUM_TEST)
    states_test = np.array([simulate_pendulum(th0, om0, d)
                             for th0, om0, d in zip(init_theta_test, init_omega_test, damping_test)])

    errs_oracle = rollout_relative_error(states_test, damping_test, HORIZON, oracle_step)
    a_o, b_o, c_o = fit_quadratic(errs_oracle, damping_test)
    oracle_results[seed] = (a_o, c_o)

    print(f"seed={seed}  oracle: a={a_o:.4f}  c={c_o:.4f}   "
          f"(NN: a={NN_TRAINED[seed][0]:.4f} c={NN_TRAINED[seed][1]:.4f} | "
          f"linear: a={LINEAR_OBSERVER[seed][0]:.4f} c={LINEAR_OBSERVER[seed][1]:.4f})")

print(f"\n{'=' * 78}")
print(f"AGGREGATE ACROSS {len(SEEDS)} SEEDS")
print(f"{'=' * 78}")
print(f"{'model':>20} {'a mean':>10} {'a std':>10} {'c mean':>10} {'c std':>10}")
for name, d in [('oracle (correct physics,', oracle_results),
                 (' wrong damping)', None)]:
    pass  

arr_o = np.array(list(oracle_results.values()))
arr_nn = np.array(list(NN_TRAINED.values()))
arr_lo = np.array(list(LINEAR_OBSERVER.values()))

for label, arr in [('oracle (physics, wrong d)', arr_o),
                    ('NN trained', arr_nn),
                    ('linear observer', arr_lo)]:
    print(f"{label:>26} {arr[:,0].mean():>10.4f} {arr[:,0].std():>10.4f} "
          f"{arr[:,1].mean():>10.4f} {arr[:,1].std():>10.4f}")

