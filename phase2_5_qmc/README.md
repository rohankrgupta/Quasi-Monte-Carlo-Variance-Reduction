# Phase 2.5: Coordinated BPS — behavior policy search with quasi-Monte Carlo sampling

Behavior policy search (BPS, [1]) lowers the variance of a policy's value estimate by collecting data with a different behavior policy and reweighting it. Phase 2 showed that tracking BPS's target with ROS fails. Here we coordinate BPS's samples instead: all episodes of the behavior policy run in lockstep, and at every step the episodes that are in similar states receive evenly spread action draws (Array-RQMC; randomized QMC for policy evaluation is due to Arnold et al., 2022). Each trajectory is still distributed exactly as the behavior policy, so importance sampling stays unbiased and independent repeats give valid error bars.

**The method.**
1. Learn the behavior policy q once, in simulation: by BPG as in [1], keeping the iterate whose OIS second moment is lowest on one fixed sample of π_e's own episodes (`core/onpolicy_select.py`), or from a pilot of on-policy episodes by the offline fit below (about 100× fewer episodes). The same q is reused across evaluations and conditions.
2. Run n episodes of q in lockstep. At each step, sort the running episodes by state (or by a value estimate when states are continuous) and give the episode of rank i the action uniform frac(φ(i) + shift_t), with φ the van der Corput sequence (the Halton sequence for several action dimensions) and a fresh random shift per step.
3. Estimate v(π_e) by ordinary importance sampling (OIS) or per-decision IS (PDIS).
4. When states repeat (discrete states): merge past weights, i.e. replace each episode's importance weight by the average over the episodes in the same state. This is marginalized importance sampling (Xie et al., 2019) computed online; it is exact there and adds up to 1.7×. In continuous states it is biased and is not used.

**Samplers** (`core/`): per-state QMC (a shifted van der Corput sequence per state); trajectory-level Sobol RQMC, Latin hypercube and antithetic pairs; Array-RQMC with weight-aware sorting, prefix-weight merging and Q-ordered actions; the CartPole samplers (i.i.d., ROS, Array-RQMC, in-budget BPS).

## Results

### 1. Discrete states (GridWorld): BPS works once its samples are coordinated

n·MSE/Var(g) at n = 8192, 1000 seeds (on-policy Monte Carlo = 1):

| Method | det πe-A | det πe-B | slip 0.1 | slip 0.2 | noisy trap |
|---|---|---|---|---|---|
| BPS + OIS, i.i.d. ([1]) | 0.90 | 0.29 | 0.39 | 0.57 | 0.54 |
| BPS + ROS (Phase 2) | 0.030 | 0.27 | 0.39 | 0.55 | 0.53 |
| **BPS + Array-RQMC, merged, PDIS** | **0.013** | **0.0049** | **0.204** | **0.335** | **0.364** |
| … with Q-ordered actions | 0.009 | 0.0022 | 0.191 | 0.357 | 0.359 |
| ROS-MC (on-policy) | 0.0012 | 0.0001 | 0.218 | 0.379 | 0.471 |
| *Model-based estimate, same data (exact)* | *0* | *0* | *0.163* | *0.326* | *0.292* |

Paired MSE ratios of the merged version at n = 8192, with 95% CIs (✓ means better):

| vs | det πe-A | det πe-B | slip 0.1 | slip 0.2 | noisy trap |
|---|---|---|---|---|---|
| BPS + OIS | 0.014 ✓ | 0.017 ✓ | 0.53 ✓ | 0.59 ✓ | 0.68 ✓ |
| BPS + ROS | 0.43 ✓ | 0.019 ✓ | 0.53 ✓ | 0.61 ✓ | 0.69 ✓ |
| ROS-MC | 10.4 ✗ | 45 ✗ | 0.94 ~ | 0.89 ✓ | 0.77 ✓ |

- **Against the original BPS methods:** 1.5–70× better than BPS + OIS on every domain, 1.5–54× better than BPS + ROS at n = 8192 (on πe-A, BPS + ROS is still better at small n). Unbiased (|bias|/SE ≤ 1.4).
- **Against ROS-MC:** matches or beats it per episode in the noisy domains (significant on slip 0.2 and the noisy trap). In deterministic domains ROS-MC wins by 7–20× even with Q-ordering: its rate there is n⁻², ours n⁻¹·⁸.
- **Why it works.** Coordination removes the variance of which action is drawn in each state. PDIS stops later weights from multiplying earlier rewards. Merging removes the spread of weights among episodes that reach the same state; with merging the estimate gets close to the model-based estimate on the same data (0.335 vs 0.326 on slip 0.2).
- **A target built for coordination doesn't help.** The variance left after coordination (J, `core/j_target.py`) is the variance of the doubly robust estimator with the true Q (Jiang and Li, 2016). Minimizing it gives a target within 3.5–6% of BPS's, and the measured difference is not significant (`results/j_target.md`); J predicts the measured variance within 0–10%. So coordinated PDIS behaves like DR with the true Q, without learning Q.

### 2. Continuous states (CartPole): coordinated BPS beats ROS

The ROS repository's two CartPole benchmarks with its trained policies (`core/cartpole.py`): CartPole-v0 with a softmax policy over two pushes, and its continuous-action version with a Gaussian policy. Episodes are sorted by a linear value estimate; weights are not merged. Force noise of 3 N makes the dynamics stochastic. ROS's α is tuned at each n over a grid that reaches the rates where it diverges. n·MSE/Var(g) at n = 4096, 1000 seeds, with the BPS target given (learned beforehand by BPG; `results/continuous/cartpole.md`, `cartpole_continuous.md`, Part A):

| Method | discrete, 0 N | discrete, 3 N | continuous actions, 0 N | continuous actions, 3 N |
|---|---|---|---|---|
| OS-MC | 1.00 | 1.07 | 1.02 | 1.05 |
| ROS-MC (α tuned at each n) | 0.211 | 0.302 | 0.596 | 0.605 |
| BPS + OIS, i.i.d. ([1]) | 0.251 | 0.328 | 0.409 | 0.420 |
| Array-RQMC, on-policy (MC) | 0.741 | 0.787 | 0.915 | 0.809 |
| **BPS + Array-RQMC, OIS** | **0.158** | **0.242** | **0.341** | **0.392** |
| paired ratio vs ROS-MC (95% CI) | 0.75 ✓ | 0.80 ✓ | 0.57 ✓ | 0.65 ✓ |
| paired ratio vs BPS + OIS | 0.63 ✓ | 0.74 ✓ | 0.83 ✓ | 0.93 ~ |

- **1.25–1.75× better than ROS per episode.** At equal environment steps: 1.25–1.41× with continuous actions, 1.03–1.09× with discrete actions (the BPS target lengthens episodes by about 20%). Unbiased (|bias|/SE ≤ 1.2), coverage 93–97%. At smaller n the margin is larger (discrete, 0 N: 0.44 at n = 64).
- **Both ingredients contribute:** coordination improves on BPS alone by 7–37%, while on-policy Array-RQMC alone is far behind ROS (0.74–0.92). BPS alone beats ROS with continuous actions (0.41 vs 0.60), not with discrete ones.
- **OIS beats PDIS here** (by 1.3–1.9×), the opposite of GridWorld.
- **Merging over neighbours in value order is biased** (|bias|/SE of 22–44, coverage 3–46%), so merging stays an add-on for discrete states.
- **Cost:** coordination adds 6–13% wall-clock per episode over i.i.d. BPS sampling; ROS costs 1.4–1.6× more per episode than ours (21–25 vs 13–16 µs) for its per-step gradient.
- These tables were produced with BPG's first selection rule (section 3); the revised rule selects the same four targets.

### 3. Continuous control (MuJoCo): coordinated BPS wins or ties tuned ROS

gymnasium's Hopper and HalfCheetah models stepped directly with MuJoCo (`core/hopper.py`, `core/control.py`), with linear policies trained by ARS and evaluated as Gaussian policies over 200 actions. Two quantities per robot: the discounted return, and the probability of a rare failure (the hopper falls, 0.74%; the cheetah's torso turns past 90°, 1.5%). HalfCheetah also has dynamics noise (sd 0.02 on the applied control). Sort key linear in [z, z², 1]; Halton points. n·MSE/Var(g) at n = 2048, 200 seeds, target given (`results/continuous/hopper_*.md`, `control_cheetah-*.md`):

| Method | Hopper fall | Hopper return | HalfCheetah return | HalfCheetah rare |
|---|---|---|---|---|
| action dimensions; π_e's action sd | 3; 0.75 | 3; 0.2 | 6; 0.05 | 6; 0.05 |
| OS-MC | 0.979 | 1.01 | 1.16 | 0.961 |
| ROS-MC (α tuned at each n) | 0.967 | 0.916 | 1.21 | 0.896 |
| BPS + OIS, i.i.d. ([1]) | 0.537 | q = π_e | q = π_e | q = π_e |
| Array-RQMC, on-policy (MC) | 1.03 | 0.734 | 0.908 | 0.959 |
| **BPS + Array-RQMC, OIS** | **0.466** | **0.734** | **0.908** | **0.959** |
| paired ratio vs ROS-MC, n = 32 / 256 / 2048 | 0.62 ✓ / 0.58 ~ / 0.48 ✓ | 1.09 ~ / 0.68 ✓ / 0.80 ~ | 1.04 ~ / 0.97 ~ / 0.75 ✓ | 1.21 ~ / 1.18 ~ / 1.07 ~ |
| coverage, ours / ROS | 95% / 95% | 100% / 100% | 100% / 95% | 80% / 85% |
| µs per episode, ours / ROS | 3194 / 4178 | 2006 / 2332 | 720 / 1253 | 779 / 1338 |

- **Rare failures: 2.1× better than ROS** (Hopper fall, n = 2048, significant; 1.6× at n = 32), unbiased (bias −1.0 SE) with 95% coverage. The gain is mostly BPS's (0.537 alone); coordination adds a further 13%, not significant.
- **Returns: coordination alone.** BPG finds no better q than π_e (Hopper's return is mostly +1 per step upright; HalfCheetah's π_e is nearly deterministic), so the gain is Array-RQMC's: 27% better than on-policy sampling on Hopper (significant), 1.46× better than ROS at n = 256 on Hopper and 1.34× at n = 2048 on HalfCheetah, ties elsewhere.
- **HalfCheetah rare: a tie.** Neither part has room. For this 1.5% event, 95% intervals from 10 runs under-cover for both methods.
- **Faster per episode on every task** (0.58–0.86× ROS's time). Hopper fall's times come from a run partly on battery power, which slowed every method alike.
- **Selecting BPG's iterate on π_e's episodes.** BPG keeps the iterate with the lowest OIS second moment, E_q[(g w)²] = E_πe[g² π_e/q], measured on one fixed sample of 40,000 π_e episodes. Measured instead on each iterate's own episodes (the first rule), it can be fooled by a q that reaches a rare event by routes π_e does not take: on HalfCheetah's rare event that rule picked a target whose true variance is about 10⁴⁴× on-policy. The revised rule picks the same target on CartPole (all four tasks), Hopper return and HalfCheetah return, whose tables were produced under the first rule, and a better one on Hopper fall (0.548 vs 0.756 on held-out π_e episodes).

### 4. The target is learned once and reused

**Offline fit** (`core/offline_bps.py`). BPG uses each episode for one gradient step; a good CartPole target took it 3.2M episodes. Instead we collect one on-policy pilot and fit q to it: minimize the pilot's importance-sampling estimate of q's variance, (1/n)Σ g_i² π_e(τ_i)/q(τ_i), plus a KL pull toward π_e, by L-BFGS over the policy's output layer only. [1] tried reusing earlier batches this way and found it unstable; the restriction to the output layer and the KL term are what make it work. Quality = Var_q(OIS)/Var(g), on 10⁶ fresh episodes (`results/continuous/offline_target.md`):

| Pilot episodes | 256 | 512 | 1024 | 2048 | BPG, 3.2M episodes |
|---|---|---|---|---|---|
| Quality of the fitted target | 1.15 | 0.49 | 0.39 | 0.27 | 0.26 |

Fits take 2–17 s. Fitting the whole network overfits at every pilot size, and below 512 episodes the fit is no better than π_e. In the exact GridWorld check, fits from small pilots without the KL term have true variances of 10³–10⁶⁰× on-policy that the empirical MSE does not show, so the shrinkage is not optional.

**Transfer.** A target learned under one level of dynamics noise is as good as one learned natively under another:

| Target | used at 0 N | used at 3 N | used at 6 N |
|---|---|---|---|
| BPG, learned at 0 N | 0.257 | 0.329 | 0.497 |
| BPG, learned at 3 N | 0.265 | 0.331 | 0.487 |
| offline fit, 2048 pilot episodes at 0 N | 0.268 | 0.337 | 0.499 |
| offline fit, 2048 pilot episodes at 3 N | 0.288 | 0.352 | 0.502 |

Quality falls with noise only because more environment noise leaves less for any behavior policy to remove.

### 5. Smooth continuous actions: QMC beats ROS and PROPS

Linear-quadratic control with a Gaussian policy (`core/lq.py`), where the value is exact. With continuous actions, the action is a smooth function of its uniform, which is where QMC is strongest.

Paired MSE ratios at n = 4096, 1000 seeds (✓ means QMC is better):

| Configuration | Sobol vs ROS | Sobol vs PROPS | Array-RQMC vs ROS |
|---|---|---|---|
| 1-D, H = 10, no environment noise | **0.0054 ✓** | **0.0015 ✓** | **0.011 ✓** |
| 1-D, H = 10, environment noise σ_w = 0.1 | **0.57 ✓** | **0.26 ✓** | **0.56 ✓** |
| 1-D, H = 10, σ_w = 0.3 | 0.94 ~ | **0.82 ✓** | 0.99 ~ |
| 1-D, H = 50, σ_w = 0.1 | 0.93 ~ | **0.21 ✓** | 0.89 ~ |
| 4-D, H = 50, σ_w = 0.1 | 0.96 ~ | **0.94 ✓** | 0.95 ~ |

- When the agent's own noise dominates, Sobol RQMC converges at n⁻². It is up to 185× better than ROS and 670× better than PROPS.
- The advantage shrinks as environment noise grows. In the 4-D, H = 50 setting no method gets far below i.i.d. sampling.
- QMC's gain over i.i.d. sampling on LQR is Arnold et al.'s result; the comparison with ROS and PROPS is the part added here.

### 6. Unbiased estimates with valid error bars

Data collected with ROS or PROPS give biased Monte Carlo estimates, because each action depends on all earlier data. When their variance is small, that bias makes confidence intervals undercover. The RQMC designs keep every trajectory exactly distributed as the sampling policy.

| | bias/SE | coverage of 95% intervals from 10 independent runs |
|---|---|---|
| ROS, deterministic GridWorld | −12 to −20 | 55–78% |
| ROS / PROPS, LQ (1-D, H = 10, σ_w ≤ 0.1) | −5 to −14 | 76–92% |
| ROS, CartPole with discrete actions (tuned α) | −4.5 to −10.6 | 79–92% |
| ROS, continuous actions (CartPole, MuJoCo) | within ±1.4 | 85–100% |
| Sobol RQMC and Array-RQMC (GridWorld, LQ) | within ±2 except one cell (+3.1) | 88–99% |
| Coordinated BPS (CartPole, MuJoCo) | within ±1.7 | 93–100%, except 80% on HalfCheetah's 1.5% event (ROS: 85%) |

With continuous actions ROS is unbiased too, so there the case for coordinated BPS is accuracy and speed, not validity.

### Where each method wins
- **Deterministic discrete domains, on-policy:** ROS. Its adaptive per-state balancing reaches n⁻² there.
- **Noisy or continuous-state domains, when the behavior policy can be learned once:** coordinated BPS. In MuJoCo it wins on rare failures (through BPS) and on smooth returns with few action dimensions (through coordination), and ties ROS when π_e is nearly deterministic.
- **Smooth continuous actions:** QMC, even on-policy.
- **Small budgets with no pilot and no logged data:** ROS; learning the target inside a few thousand episodes does not pay.

**Tried, neutral or negative:**
- Per-state QMC is a weaker, non-adaptive ROS: with ROS's α tuned at each n, it wins only on πe-A at n ≤ 256. With BPS's weights it blows up.
- Latin hypercube and antithetic sampling are weaker than Sobol. Antithetic is 2× worse than i.i.d. in LQ, whose returns are symmetric in the action noise.
- Adding RQMC on top of ROS gives nothing.
- On-policy in the discrete GridWorld, ROS beats every QMC sampler.
- Per environment step in GridWorld, coordinated BPS loses to ROS-MC, because BPS's target lengthens episodes (25–27 steps vs 19–21).
- Q-ordering used the exact Q (a best case); in practice Q would have to be estimated.
- Weight merging over nearest neighbours in continuous states is biased, and the bias does not shrink with n.
- The cross-entropy target (|return|-weighted behavior cloning of π_e's own episodes) equals the BPS target within 1–2% in the exact GridWorld check, as [1] found for BPG-KL, but from small pilots it is worse than the offline fit.

## Run

From the repository root:

```bash
python -m phase2_5_qmc.experiments.gridworld                   # 6 min: QMC samplers vs ROS on the GridWorld (+ figure)
python -m phase2_5_qmc.experiments.array_rqmc                  # 4 min: Array-RQMC, bias and interval coverage
python -m phase2_5_qmc.experiments.j_target                    # 8 min: a target designed for coordination
python -m phase2_5_qmc.experiments.merging                     # 8 min: prefix-weight merging and Q-ordered actions
python -m phase2_5_qmc.experiments.continuous.lq               # 1 hour: continuous actions, QMC vs ROS vs PROPS
python -m phase2_5_qmc.experiments.continuous.cartpole discrete    # 45 min: CartPole, discrete actions
python -m phase2_5_qmc.experiments.continuous.cartpole continuous  # 45 min: CartPole, continuous actions
python -m phase2_5_qmc.experiments.continuous.offline_target   # 6 min: the offline target fit and its transfer
python -m phase2_5_qmc.experiments.continuous.hopper fall      # 2.5 hours: Hopper, probability of falling within 200 actions
python -m phase2_5_qmc.experiments.continuous.hopper return    # 2.5 hours: Hopper, discounted return over 200 actions
python -m phase2_5_qmc.experiments.continuous.control cheetah-return  # 1 hour: HalfCheetah, discounted return
python -m phase2_5_qmc.experiments.continuous.control cheetah-rare    # 1 hour: HalfCheetah, torso past 90° within 200 actions
```

## Code

```
core/qmc.py            per-state QMC and trajectory-level designs (Sobol, Latin hypercube, antithetic)
core/array_rqmc.py     Array-RQMC in lockstep, with weight-aware sorting, prefix-weight merging, Q-ordering (tabular)
core/j_target.py       the variance left after coordination (J), its exact gradient and its minimizer
core/lq.py             linear-quadratic testbed (exact value) and its samplers, including ROS and PROPS
core/cartpole.py       the ROS repo's CartPole tasks and policies: i.i.d., ROS, Array-RQMC and in-budget BPS samplers
core/offline_bps.py    the offline fit of the BPS target from an on-policy pilot
core/hopper.py         gymnasium's Hopper-v5 stepped directly with MuJoCo, a linear Gaussian policy (ARS), and its samplers
core/control.py        HalfCheetah (and Reacher) stepped with MuJoCo, with dynamics noise; ARS policies, samplers and BPG
core/onpolicy_select.py  BPG's iterate selection: each iterate's OIS second moment measured on π_e's own episodes
```

Environments, exact DP, ROS toward πe and BPS's target come from Phases 1 and 2.

**References.**
- S. M. R. Arnold, P. L'Ecuyer, L. Chen, Y. Chen, F. Sha. *Policy Learning and Evaluation with Randomized Quasi-Monte Carlo.* AISTATS 2022.
- P. L'Ecuyer, C. Lécot, B. Tuffin. *A Randomized Quasi-Monte Carlo Simulation Method for Markov Chains.* Operations Research 2008.
- D. Precup, R. S. Sutton, S. Singh. *Eligibility Traces for Off-Policy Policy Evaluation.* ICML 2000 (per-decision IS).
- T. Xie, Y. Ma, Y.-X. Wang. *Towards Optimal Off-Policy Evaluation for Reinforcement Learning with Marginalized Importance Sampling.* NeurIPS 2019.
- N. Jiang, L. Li. *Doubly Robust Off-Policy Value Evaluation for Reinforcement Learning.* ICML 2016.
- N. E. Corrado, J. P. Hanna. *On-Policy Policy Gradient Reinforcement Learning Without On-Policy Sampling* (PROPS).
