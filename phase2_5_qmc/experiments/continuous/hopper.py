"""Continuous control: coordinated BPS on Hopper (gymnasium's Hopper-v5, stepped directly; core/hopper.py).

pi_e: a linear policy trained by ARS (train_ars, seed 0, full 1000-step episodes; cached in results/continuous/
hopper_ars.npz), evaluated as a Gaussian policy with action noise sd sigma over HORIZON = 200 actions. Two settings:
  "fall":   the probability that the hopper falls within 200 actions (sigma = 0.75; about 0.8%): a rare event;
  "return": Hopper-v5's discounted return (sigma = 0.2, gamma = 0.99); returns vary little relative to their mean.
Ground truth: N_TRUTH on-policy episodes. n = 32 / 256 / 2048 episodes, 200 seeds; ROS's learning rate is tuned at
each n on 100 other seeds.

BPS target given: a fixed target learned beforehand by BPG (bpg_target; its episodes are not counted), as in Part A of
the CartPole experiments. Methods: OS-MC; ROS-MC; BPS + OIS and BPS + PDIS (i.i.d.); on-policy Array-RQMC (MC);
BPS + Array-RQMC with OIS and with PDIS. Weights are not merged (continuous states).

Reported: n·MSE/Var(g) per episode and at equal environment steps (on-policy MC = 1), bias/SE, coverage and width of
95% t-intervals from groups of 10 runs, wall-clock time per episode at the largest n (10 processes), paired MSE ratios.

Run:    python -m phase2_5_qmc.experiments.continuous.hopper fall      (about 2 hours)
        python -m phase2_5_qmc.experiments.continuous.hopper return    (about 2 hours)
Output: results/continuous/hopper_fall.md, results/continuous/hopper_return.md
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from phase1_reproduction.core import stats
from phase2_5_qmc.core import hopper as hp
from phase2_5_qmc.experiments.array_rqmc import ci_stats
from phase2_5_qmc.experiments.gridworld import mark, md

OUT = Path(__file__).resolve().parents[2] / "results" / "continuous"
NS = (32, 256, 2048)
N_EVAL, N_TUNE = 200, 100
N_TRUTH = 400_000
BPG = dict(iters=100, batch=1000, step=0.01)
CONFIGS = {   # ROS grids span the tuned rate on both sides (ROS diverges at 1e-2: the linear mean is unbounded)
    "fall": dict(sigma=0.75, task="fall", ros_lrs=(1e-5, 3e-5, 1e-4, 3e-4)),
    "return": dict(sigma=0.2, task="return", ros_lrs=(1e-4, 3e-4, 1e-3, 3e-3))}
METHODS = ("OS-MC", "ROS-MC", "BPS + OIS (i.i.d.)", "BPS + PDIS (i.i.d.)", "Array-RQMC, on-policy (MC)",
           "BPS + Array-RQMC, OIS", "BPS + Array-RQMC, PDIS")
CONTRASTS = (("BPS + Array-RQMC, OIS", "BPS + OIS (i.i.d.)"), ("BPS + Array-RQMC, PDIS", "BPS + PDIS (i.i.d.)"),
             ("BPS + Array-RQMC, OIS", "ROS-MC"), ("BPS + Array-RQMC, PDIS", "ROS-MC"),
             ("Array-RQMC, on-policy (MC)", "OS-MC"), ("Array-RQMC, on-policy (MC)", "ROS-MC"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _seed_run(args):
    kind, pi_e, other, n, seed, stream, value_w, *task = args
    task = task[0] if task else "return"
    if kind == "ros":
        mc, length = hp.run_ros(pi_e, other, n, seed, stream, task)
        return {"MC": mc, "length": length}
    mc, ois, pdis, length = hp.run(pi_e, other, n, seed, stream, kind, value_w, task)
    return {"MC": mc, "OIS": ois, "PDIS": pdis, "length": length}


def ars_policy():
    """The ARS-trained linear policy (trained once, then read from results/continuous/hopper_ars.npz)."""
    path = OUT / "hopper_ars.npz"
    if not path.exists():
        W, mean, std, ret = hp.train_ars()
        np.savez(path, W=W, mean=mean, std=std, ret=ret)
    z = np.load(path)
    return z["W"], z["mean"], z["std"], float(z["ret"])


def main(setting="fall"):
    cfg = CONFIGS[setting]
    SIGMA, task = cfg["sigma"], cfg["task"]
    OUT.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("spawn").Pool(10)
    W, mean, std, ret = ars_policy()
    pi_e = hp.Policy(W, np.zeros(hp.ACT_DIM), np.full(hp.ACT_DIM, np.log(SIGMA)), mean, std)
    log(f"{setting}: ARS policy (mean return {ret:.0f} on 1000-step episodes), sigma {SIGMA}")

    truth = pool.map(_seed_run, [("iid", pi_e, None, N_TRUTH // 400, s, "truth", None, task) for s in range(400)])
    mc = np.array([r["MC"] for r in truth])
    v = mc.mean()
    if task == "fall":                          # an indicator: Var(g) = p (1 - p)
        var = v * (1 - v)
    else:                                       # from 20000 single episodes (the truth runs only return means)
        singles = pool.map(_seed_run, [("iid", pi_e, None, 1, s, "variance", None, task) for s in range(20000)])
        var = np.var([r["MC"] for r in singles], ddof=1)
    L_e = np.mean([r["length"] for r in truth])
    vw = hp.fit_value(pi_e, map_fn=pool.map, task=task)
    t0 = time.perf_counter()
    q = hp.bpg_target(pi_e, map_fn=pool.map, task=task, **BPG)
    t_bpg = time.perf_counter() - t0
    log(f"v = {v:.5g} ± {mc.std(ddof=1) / np.sqrt(len(mc)):.2g}, Var(g) = {var:.4g}, length {L_e:.1f}; "
        f"BPS target learned in {t_bpg:.0f} s")

    memo, methods = {}, {}

    def runs(kind, other, n, k, stream):
        """Per-seed results of k seeds, run in parallel (memoized), and their wall-clock time."""
        tag = (kind, id(other) if not np.isscalar(other) else other, n, k, stream)
        if tag not in memo:
            t = time.perf_counter()
            out = pool.map(_seed_run, [(kind, pi_e, other, n, s, stream, vw, task) for s in range(k)])
            memo[tag] = ({key: np.array([o[key] for o in out]) for key in out[0]}, time.perf_counter() - t)
        return memo[tag]

    def evaluate(name, kind, other, key, grid=None):
        est, length, chosen = [], [], []
        for n in NS:
            h = other
            if grid is not None:
                h = min(grid, key=lambda x: np.nan_to_num(np.mean((runs(kind, x, n, N_TUNE, "tune")[0][key] - v) ** 2), nan=np.inf))
            r, sec = runs(kind, h, n, N_EVAL, "eval")
            est.append(r[key])
            length.append(r["length"].mean())
            chosen.append(h if grid is not None else None)
        methods[name] = {"est": np.stack(est, 1), "length": np.array(length), "chosen": chosen,
                         "us": 1e6 * sec / (N_EVAL * NS[-1])}
        log(f"  {name}")

    evaluate("OS-MC", "iid", None, "MC")
    evaluate("ROS-MC", "ros", None, "MC", cfg["ros_lrs"])
    evaluate("BPS + OIS (i.i.d.)", "iid", q, "OIS")
    evaluate("BPS + PDIS (i.i.d.)", "iid", q, "PDIS")
    evaluate("Array-RQMC, on-policy (MC)", "array", None, "MC")
    evaluate("BPS + Array-RQMC, OIS", "array", q, "OIS")
    evaluate("BPS + Array-RQMC, PDIS", "array", q, "PDIS")
    pool.close()

    notes = ""
    if np.array_equal(q.theta(), pi_e.theta()):
        notes += "No BPG iterate beat pi_e on validation, so q = pi_e and the BPS rows equal the on-policy rows. "
    if task == "fall":
        notes += "OIS = PDIS: the only reward is at an episode's last step. "
    what = "probability of falling within" if task == "fall" else "discounted return over"
    text = [f"# Coordinated BPS in continuous control: Hopper, {what} {hp.HORIZON} actions\n",
            f"pi_e: ARS linear policy (mean return {ret:.0f} on full 1000-step episodes) with action noise sd {SIGMA}. "
            f"Truth v = {v:.5g} ± {mc.std(ddof=1) / np.sqrt(len(mc)):.2g} "
            f"({N_TRUTH} episodes), Var(g) = {var:.4g}, mean length {L_e:.1f}. BPS target learned offline by BPG in "
            f"{t_bpg:.0f} s (not counted; iterates compared on 40000 episodes of pi_e). {notes}n·MSE/Var(g): "
            f"on-policy MC = 1; {N_EVAL} seeds. Coverage: 95% t-intervals from groups of 10 runs. Time: µs per episode at n = {NS[-1]} "
            "on 10 processes.\n"]
    width = ci_stats(methods["OS-MC"]["est"][:, -1], v, 1.0)[1]
    rows = []
    for method in METHODS:
        m = methods[method]
        y = np.array(NS) * ((m["est"] - v) ** 2).mean(0) / var
        e = m["est"][:, -1]
        cov, w = ci_stats(e, v, width)
        rows.append([method, *(f"{x:.3g}" for x in y), f"{y[-1] * m['length'][-1] / L_e:.3g}",
                     f"{(e.mean() - v) / (e.std(ddof=1) / np.sqrt(len(e))):+.1f}", f"{cov:.0%}", f"{w:.2f}",
                     f"{m['us']:.0f}", ", ".join(f"{c:g}" for c in m["chosen"]) if m["chosen"][0] else "–"])
    text.append(md(["method", *(f"n = {n}" for n in NS), f"equal steps, n = {NS[-1]}", "bias/SE", "coverage", "width",
                    "µs/episode", "tuned (per n)"], rows) + "\n")
    contrasts = []
    for x, y in CONTRASTS:
        cells = []
        for j in range(len(NS)):
            r, lo, hi = stats.paired_mse_ratio(methods[x]["est"][:, j], methods[y]["est"][:, j], v)
            cells.append(f"{r:.3g} {mark(lo, hi)}")
        contrasts.append([f"{x} vs {y}", *cells])
    text.append("## Paired MSE ratios per episode\n\n✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).\n\n"
                + md(["comparison", *(f"n = {n}" for n in NS)], contrasts) + "\n")
    (OUT / f"hopper_{setting}.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main(*sys.argv[1:])
