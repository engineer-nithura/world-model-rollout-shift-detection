
#Large-angle pendulum: theta'' = -(g/L) sin(theta) - damping * theta'


import numpy as np
from sklearn.neural_network import MLPRegressor

SEED = 0
np.random.seed(SEED)

#Config
LATENT_DIM = 4
HORIZON = 20
NUM_TRAIN = 3000
NUM_TEST = 1500
HIDDEN_DIM = 128
TRAJ_STEPS = 40
DT = 0.05
G_OVER_L = 9.8   # g/L, using L=1
CLIP_MAG = 1e3
N_BINS = 9
EPS = 1e-3
THETA0_RANGE = 2.2   # large swing angle, radians (~126 degrees) -- where sin(theta) != theta
OMEGA0_RANGE = 1.0


#Physics function
def simulate_pendulum(theta0, omega0, damping, g_over_l=G_OVER_L, dt=DT, steps=TRAJ_STEPS):
    states = np.zeros((steps, 2))
    theta, omega = theta0, omega0
    for t in range(steps):
        states[t] = [theta, omega]
        alpha = -g_over_l * np.sin(theta) - damping * omega
        omega += alpha * dt
        theta += omega * dt
    return states


print("Generating data (large-angle pendulum)...")
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

assert np.all(np.isfinite(states_train)) and np.all(np.isfinite(states_test)), \
    "Ground-truth pendulum trajectories contain NaN/Inf -- reduce dt or THETA0_RANGE."

# Quantify how much the nonlinearity actually diverges from the linear approximation
sample_thetas = states_train[:, :, 0].flatten()
sin_vs_linear_gap = np.abs(np.sin(sample_thetas) - sample_thetas)
print(f"  theta range covered: [{sample_thetas.min():.3f}, {sample_thetas.max():.3f}] rad")
print(f"  mean |sin(theta) - theta| over training data: {sin_vs_linear_gap.mean():.4f} "
      f"(0 would mean the small-angle approximation is exact)")

X_train = states_train[:, :-1, :].reshape(-1, 2)
y_train = states_train[:, 1:, :].reshape(-1, 2)


# Neural net (approximation of world model)
def nn_forward(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    z = np.tanh(h1 @ W2 + b2)
    pred = z @ W3 + b3
    return pred

print(f"\nTraining neural world model (latent_dim={LATENT_DIM})...")
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
nn_train_pred = nn_forward(X_train, *W_trained)
nn_train_err = np.mean(np.linalg.norm(nn_train_pred - y_train, axis=1))
print(f"  NN mean train-set one-step error: {nn_train_err:.5f}")

rng = np.random.RandomState(123)
W1_rand = rng.uniform(-1, 1, (2, HIDDEN_DIM)) / np.sqrt(2)
b1_rand = np.zeros(HIDDEN_DIM)
W2_rand = rng.uniform(-1, 1, (HIDDEN_DIM, LATENT_DIM)) / np.sqrt(HIDDEN_DIM)
b2_rand = np.zeros(LATENT_DIM)
W3_rand = rng.uniform(-1, 1, (LATENT_DIM, 2)) / np.sqrt(LATENT_DIM)
b3_rand = np.zeros(2)
W_untrained = (W1_rand, b1_rand, W2_rand, b2_rand, W3_rand, b3_rand)


# Classical linear observer
def fit_linear_observer(X_train, y_train):
    A_T, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return A_T

A_T = fit_linear_observer(X_train, y_train)
lo_train_pred = X_train @ A_T
lo_train_err = np.mean(np.linalg.norm(lo_train_pred - y_train, axis=1))
print(f"\nFitting linear observer (least squares)...")
print(f"  fitted A^T =\n{A_T}")
print(f"  linear observer mean train-set one-step error: {lo_train_err:.5f}")
print(f"  (compare to NN's {nn_train_err:.5f} -- if linear observer's error is clearly")
print(f"   larger, the nonlinearity is finally biting, unlike the quadratic-drag system)")


# Closed-loop mean RELATIVE rollout error
def rollout_relative_error(predict_fn):
    mean_rel_err = np.zeros(states_test.shape[0])
    for i in range(states_test.shape[0]):
        true_points = states_test[i, 0:HORIZON + 1]
        current_state = true_points[0]
        step_errs = []
        diverged = False
        for step in range(HORIZON):
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


def bin_and_fit(errs, label):
    bin_edges = np.linspace(0.1, 0.9, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(damping_test, bin_edges) - 1, 0, N_BINS - 1)

    print(f"\n  [{label}]")
    print(f"  {'damping bin':>14} {'n':>6} {'mean rel err':>14}")
    bin_means = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_means[b] = errs[mask].mean()
        print(f"  {bin_centers[b]:>14.3f} {mask.sum():>6} {bin_means[b]:>14.4f}")

    valid = ~np.isnan(bin_means)
    x = bin_centers[valid] - 0.5
    y = bin_means[valid]
    a, b_coef, c = np.polyfit(x, y, 2)
    print(f"  -> quadratic fit: a(curvature)={a:>8.4f}  b(asymmetry)={b_coef:>8.4f}  c(baseline@0.5)={c:>8.4f}")
    return a, b_coef, c


print(f"\n{'=' * 70}")
print(f"Closed-loop relative error, horizon={HORIZON}, large-angle pendulum")
print(f"{'=' * 70}")

errs_nn_trained = rollout_relative_error(lambda s: nn_forward(s, *W_trained))
a_nn_t, b_nn_t, c_nn_t = bin_and_fit(errs_nn_trained, "NN trained")

errs_nn_untrained = rollout_relative_error(lambda s: nn_forward(s, *W_untrained))
a_nn_u, b_nn_u, c_nn_u = bin_and_fit(errs_nn_untrained, "NN untrained")

errs_lo = rollout_relative_error(lambda s: s @ A_T)
a_lo, b_lo, c_lo = bin_and_fit(errs_lo, "linear observer")

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"{'model':>20} {'a(curvature)':>14} {'c(baseline@0.5)':>18}")
print(f"{'NN trained':>20} {a_nn_t:>14.4f} {c_nn_t:>18.4f}")
print(f"{'NN untrained':>20} {a_nn_u:>14.4f} {c_nn_u:>18.4f}")
print(f"{'linear observer':>20} {a_lo:>14.4f} {c_lo:>18.4f}")
