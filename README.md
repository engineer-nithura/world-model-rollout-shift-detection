# World Model Rollout Shift Detection

Testing whether a single trained world model can detect its own distribution shift using only closed-loop rollout error using no ensembles or labels. Calibrated against a physics-oracle ceiling (~57% recovery) and shown to diverge from task-regret, a standard curriculum-learning difficulty signal.

## Core Claim

A world model trained only on one setting of a hidden environment parameter (pendulum damping = 0.5) produces closed-loop rollout error that grows measurably and systematically as true damping moves away from that value — a self-contained, label-free signal for detecting distribution shift. This is validated against an untrained-network control (rules out architecture artifacts), a physics-oracle ceiling (calibrates how much of the best-possible signal is recovered), and shown to diverge from a regret-based task-difficulty signal (motivating it as a distinct signal for curriculum-learning applications).

Key metrics used throughout, computed from binned relative rollout error as a function of damping:
- **Curvature (a)**: how sharply error rises with distance from the training value (detection sensitivity).
- **Baseline error (c)**: error at the training value itself (in-regime fit quality).
- **Regret**: task cost under a fixed controller minus task cost under a per-trajectory optimal controller, used to test whether rollout error captures something distinct from task-performance-based difficulty.

## Repository Structure

```
code/
├── pendulum_large_angle_comp.py
├── pendulum_seed_robustness_check.py
├── oracle_and_linear_baseline.py
├── pendulum_bottleneck_sweep.py
├── nonlinear_damping_rollout_test.py
├── curriculum_divergence_seed_check.py
└── curriculum_divergence_pd_controller.py
figures/
```

This repo contains the code and figures behind the project; the full paper draft is maintained separately and is not included here.

## Requirements

```
numpy
scikit-learn
scipy
matplotlib
```

No GPU required — all models are small MLPs (128-unit hidden layer, 2–16 unit bottleneck) trained with scikit-learn's `MLPRegressor`.

## Scripts, in Run Order

Each script is self-contained (regenerates its own data, no shared saved state between scripts) and can be run independently. The numeric prefixes reflect a logical dependency order, not a strict execution requirement — with one exception noted below.

**`pendulum_large_angle_comp.py`**
Single-run demonstration of the core effect on the primary system (large-angle pendulum). Trains one world model at damping=0.5, evaluates closed-loop rollout error across damping 0.1–0.9 against an untrained control and a least-squares linear observer. Produces the main U-shaped error curve (Figure 1) and the bin-level data behind it.

**`pendulum_seed_robustness_check.py`**
Repeats the setup above across 5 independently seeded trainings (data generation + model init jointly re-seeded), reporting curvature (`a`) and baseline error (`c`) as mean ± std for trained model, untrained control, and linear observer. Produces Table 1.

**`oracle_and_linear_baseline.py`**
Constructs the physics-oracle baseline (exact nonlinear dynamics, same damping uncertainty as the trained model) and evaluates it across the same 5 seeds as `pendulum_seed_robustness_check.py`. Reports trained-model and linear-observer recovery as a percentage of the oracle ceiling. Produces Table 2 / Figure 2.
*Note:* this script hardcodes the `NN_TRAINED` and `LINEAR_OBSERVER` per-seed values from an actual run of `pendulum_seed_robustness_check.py` rather than recomputing them, to avoid re-running training just to add the oracle comparison. If you rerun that script and get different numbers (e.g. different sklearn version, different seed behavior), update the hardcoded dictionaries at the top of this file to match before trusting its output.

**`pendulum_bottleneck_sweep.py`**
Reruns the core effect at bottleneck sizes k ∈ {2, 4, 8, 16}, single seed each, on the primary pendulum system. Identifies a minimum-capacity threshold (k=2 fails to fit the training regime) and a mild capacity-curvature trend above it. Produces Table 3 / Figure 3.

**`nonlinear_damping_rollout_test.py`**
Reruns the core effect on a second, independent nonlinear system (quadratic velocity-dependent drag) to confirm the effect isn't specific to the pendulum's particular nonlinearity. Single seed.

**`curriculum_divergence_seed_check.py`**
Builds a fixed single-gain damping controller and a per-trajectory grid-searched optimal controller (same functional form, true damping known), computes regret, and correlates it against rollout error across 5 seeds. Produces Table 4 (single-gain row).

**`curriculum_divergence_pd_controller.py`**
Repeats the regret-divergence test above with a structurally different controller (two-gain PD control) to check the result isn't specific to one controller family. Single seed. Produces Table 4 (PD row).

## Reproducibility Notes

- All random seeds are explicit (`numpy.random.RandomState` with logged offsets per data stream) — no global random state is relied on implicitly.
- `pendulum_seed_robustness_check.py`, `oracle_and_linear_baseline.py`, and `curriculum_divergence_seed_check.py` replicate across 5 seeds; `pendulum_large_angle_comp.py`, `pendulum_bottleneck_sweep.py`, `nonlinear_damping_rollout_test.py`, and `curriculum_divergence_pd_controller.py` are single-seed and are reported as such in the paper, not folded into the multi-seed claims.
- Grid search resolution for the regret metric (`curriculum_divergence_seed_check.py`, `curriculum_divergence_pd_controller.py`) is fixed at 151 points (single-gain) / 9×16 (PD), chosen to keep worst-case discretization error small. Correctness is checked directly: the fixed controller's own gain is included as a candidate in the search grid, so regret ≥ 0 is guaranteed by construction and verified empirically at runtime (~0% negative-regret trajectories in each run).

## What's Not in This Repo

An earlier pipeline (teacher-forced residual regression on raw/latent/residual representations) was tested and ruled out early in this project — an untrained-model control showed the observed effect didn't require training at all. That code is not included here since none of its results are part of the current findings; it's mentioned for transparency about what was tried and discarded before arriving at the closed-loop rollout approach used throughout this repo.

## Status

This is a preliminary, single-author research project shared for early feedback and potential collaboration — not a finished submission. Open questions include generalization to learned/adaptive controllers (the curriculum-divergence scripts currently use hand-tuned fixed controllers as a task-difficulty proxy, not a trained RL policy) and to higher-dimensional systems.
