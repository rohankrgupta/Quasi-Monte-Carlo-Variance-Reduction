"""Coordinated BPS in continuous control (core/control.py): a smooth return and a rare failure on the same policy.

The 2x2 ablation: on each domain, one task with a smooth return (where coordination is expected to help) and one
rare failure event (where BPS is expected to help, by sampling the failure more often). Does BPS + coordination beat
each alone on either?

pi_e: a linear policy trained by ARS (train_ars, seed 0, noise-free dynamics; cached in results/continuous/
control_ars_<domain>.npz), run as a Gaussian policy with action noise sd sigma, under dynamics noise env_sd.
Ground truth: N_TRUTH on-policy episodes. n = 32 / 256 / 2048 episodes, N_EVAL seeds; ROS's learning rate is tuned per
n on N_TUNE other seeds over a grid that reaches rates where ROS diverges.

BPS target given: learned beforehand by BPG (bpg_target; its episodes are not counted). Methods: OS-MC; ROS-MC;
BPS + OIS and BPS + PDIS (i.i.d.); on-policy Array-RQMC (MC); BPS + Array-RQMC with OIS and with PDIS.

Reported: n·MSE/Var(g) per episode and at equal environment steps (on-policy MC = 1), bias/SE, coverage and width of
95% t-intervals from groups of 10 runs, wall-clock time per episode at the largest n (10 processes), paired MSE ratios.

Run:    python -m phase2_5_qmc.experiments.continuous.control cheetah-return    (about 2.5 hours)
        python -m phase2_5_qmc.experiments.continuous.control cheetah-rare      (about 2.5 hours)
Output: results/continuous/control_<setting>.md
"""
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from phase1_reproduction.core import stats
from phase2_5_qmc.core import control as ct
from phase2_5_qmc.experiments.array_rqmc import ci_stats
from phase2_5_qmc.experiments.gridworld import mark, md

OUT = Path(__file__).resolve().parents[2] / "results" / "continuous"
NS = (32, 256, 2048)
N_EVAL, N_TUNE, N_TRUTH = 200, 100, 400_000
BPG = dict(iters=100, batch=1000, step=0.01)
SETTINGS = {   # sigma: action noise sd; ros_lrs reach rates where ROS diverges (1e-2 on HalfCheetah)
    "cheetah-return": dict(task=ct.Task("cheetah", 0.02, "return"), sigma=0.05, ros_lrs=(3e-5, 1e-4, 3e-4, 1e-3)),
    "cheetah-rare": dict(task=ct.Task("cheetah", 0.02, "rare", math.pi / 2), sigma=0.05,
                         ros_lrs=(3e-5, 1e-4, 3e-4, 1e-3)),
}
METHODS = ("OS-MC", "ROS-MC", "BPS + OIS (i.i.d.)", "BPS + PDIS (i.i.d.)", "Array-RQMC, on-policy (MC)",
           "BPS + Array-RQMC, OIS", "BPS + Array-RQMC, PDIS")
CONTRASTS = (("BPS + Array-RQMC, OIS", "BPS + OIS (i.i.d.)"), ("BPS + Array-RQMC, OIS", "Array-RQMC, on-policy (MC)"),
             ("BPS + Array-RQMC, OIS", "ROS-MC"), ("BPS + Array-RQMC, PDIS", "ROS-MC"),
             ("BPS + OIS (i.i.d.)", "OS-MC"), ("Array-RQMC, on-policy (MC)", "OS-MC"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _seed_run(args):
    kind, task, pi_e, other, n, seed, stream, value_w = args
    if kind == "ros":
        mc, length = ct.run_ros(task, pi_e, other, n, seed, stream)
        return {"MC": mc, "length": length}
    mc, ois, pdis, length = ct.run(task, pi_e, other, n, seed, stream, kind, value_w)
    return {"MC": mc, "OIS": ois, "PDIS": pdis, "length": length}


def ars_policy(domain):
    """The domain's ARS-trained linear policy (trained once, then read from results/continuous/)."""
    path = OUT / f"control_ars_{domain}.npz"
    if not path.exists():
        horizon = 1000 if domain == "cheetah" else None          # HalfCheetah's full episodes
        W, mean, std, ret = ct.train_ars(domain, horizon=horizon)
        np.savez(path, W=W, mean=mean, std=std, ret=ret)
    z = np.load(path)
    return z["W"], z["mean"], z["std"], float(z["ret"])


def main(setting="cheetah-return"):
    cfg = SETTINGS[setting]
    task, sigma = cfg["task"], cfg["sigma"]
    OUT.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("spawn").Pool(10)
    W, mean, std, ret = ars_policy(task.domain)
    k = W.shape[0]
    pi_e = ct.Policy(W, np.zeros(k), np.full(k, np.log(sigma)), mean, std)
    log(f"{setting}: ARS policy (return {ret:.0f} on its training episodes), sigma {sigma}, dynamics noise {task.env_sd}")

    truth = pool.map(_seed_run, [("iid", task, pi_e, None, N_TRUTH // 400, s, "truth", None) for s in range(400)])
    mc = np.array([r["MC"] for r in truth])
    v, se_v = mc.mean(), mc.std(ddof=1) / np.sqrt(len(mc))
    if task.kind == "rare":                     # an indicator: Var(g) = p (1 - p)
        var = v * (1 - v)
    else:
        singles = pool.map(_seed_run, [("iid", task, pi_e, None, 1, s, "variance", None) for s in range(20000)])
        var = np.var([r["MC"] for r in singles], ddof=1)
    L_e = np.mean([r["length"] for r in truth])
    vw = ct.fit_value(task, pi_e, map_fn=pool.map)
    t0 = time.perf_counter()
    q = ct.bpg_target(task, pi_e, map_fn=pool.map, **BPG)
    t_bpg = time.perf_counter() - t0
    log(f"v = {v:.5g} ± {se_v:.2g}, Var(g) = {var:.4g}, length {L_e:.1f}; BPS target learned in {t_bpg:.0f} s")

    memo, methods = {}, {}

    def runs(kind, other, n, k, stream):
        """Per-seed results of k seeds, run in parallel (memoized), and their wall-clock time."""
        tag = (kind, id(other) if not np.isscalar(other) else other, n, k, stream)
        if tag not in memo:
            t = time.perf_counter()
            out = pool.map(_seed_run, [(kind, task, pi_e, other, n, s, stream, vw) for s in range(k)])
            memo[tag] = ({key: np.array([o[key] for o in out]) for key in out[0]}, time.perf_counter() - t)
        return memo[tag]

    def evaluate(label, kind, other, key, grid=None):
        est, length, chosen = [], [], []
        for n in NS:
            h = other
            if grid is not None:
                h = min(grid, key=lambda x: np.nan_to_num(
                    np.mean((runs(kind, x, n, N_TUNE, "tune")[0][key] - v) ** 2), nan=np.inf))
            r, sec = runs(kind, h, n, N_EVAL, "eval")
            est.append(r[key])
            length.append(r["length"].mean())
            chosen.append(h if grid is not None else None)
        methods[label] = {"est": np.stack(est, 1), "length": np.array(length), "chosen": chosen,
                          "us": 1e6 * sec / (N_EVAL * NS[-1])}
        log(f"  {label}")

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
    what = (f"probability that the torso turns past {task.threshold:.3g} rad within" if task.kind == "rare"
            else "discounted return over")
    env = task.env(1)
    text = [f"# Coordinated BPS in continuous control: {type(env).__name__}, {what} {env.horizon} actions\n",
            f"pi_e: ARS linear policy with action noise sd {sigma}; dynamics noise sd {task.env_sd} on the applied "
            f"control. Truth v = {v:.5g} ± {se_v:.2g} ({N_TRUTH} episodes), Var(g) = {var:.4g}, mean length "
            f"{L_e:.1f}. BPS target learned offline by BPG in {t_bpg:.0f} s (not counted; iterates compared on "
            f"40000 episodes of pi_e). {notes}n·MSE/Var(g): on-policy MC = 1; {N_EVAL} seeds. Coverage: 95% t-intervals from groups of 10 runs. Time: µs per "
            f"episode at n = {NS[-1]} on 10 processes. ROS grid: {', '.join(f'{h:g}' for h in cfg['ros_lrs'])}.\n"]
    width = ci_stats(methods["OS-MC"]["est"][:, -1], v, 1.0)[1]
    rows = []
    for label in METHODS:
        m = methods[label]
        y = np.array(NS) * ((m["est"] - v) ** 2).mean(0) / var
        e = m["est"][:, -1]
        cov, w = ci_stats(e, v, width)
        rows.append([label, *(f"{x:.3g}" for x in y), f"{y[-1] * m['length'][-1] / L_e:.3g}",
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
    (OUT / f"control_{setting}.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main(*sys.argv[1:])
