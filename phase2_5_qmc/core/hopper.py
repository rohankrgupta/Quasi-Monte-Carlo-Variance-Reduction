"""Hopper (gymnasium's Hopper-v5) with a linear Gaussian evaluation policy: coordinated BPS in continuous control.

Environment: gymnasium's Hopper-v5 model file and rules, stepped with MuJoCo directly so that many episodes can run in
lockstep (with the same start state and actions it reproduces gymnasium.make("Hopper-v5") exactly). 4 MuJoCo steps
of 2 ms per action; reward = forward velocity + 1 (while healthy) - 1e-3 |a|^2; the episode ends once the hopper is
unhealthy (height below 0.7, torso angle beyond 0.2 rad, or any state beyond 100) or after HORIZON actions; the start
state is the rest pose plus uniform noise on [-0.005, 0.005]; observation = qpos[1:] and clip(qvel, -10, 10).
All episodes share one MjData: each episode's integration state (58 numbers) is swapped in before its step.

Policy: a ~ N(W z + b, diag(sd^2)) with z = (s - mean) / std, W trained by ARS (train_ars); parameters theta =
(W, b, log sd), 39 numbers, so BPS and ROS move them as one vector. Every action is mean + sd * Phi^{-1}(u) for a
uniform u in each of the 3 dimensions, so samplers differ only in how they choose the uniforms:
  * iid:   counter-based uniforms (rng.uniform at (seed, episode, step, dimension)).
  * array: Array-RQMC. At each step the running episodes are sorted by a value estimate V(s) = w . [z, z^2, 1], and
           the episode of rank i gets the shifted Hammersley point (phi_2(i), phi_3(i), phi_5(i)) + shift_t mod 1.
  * ros:   ROS as in the ROS repo, episodes one after another: before each action the behavior is
           theta_e - lr * (mean over all past steps of grad log pi_e(a|s)).
Estimates: MC (mean return), OIS and PDIS, plus the mean episode length.
Tasks: "return" (Hopper-v5's discounted return, gamma = 0.99) or "fall" (reward 1 when the hopper falls, i.e. the
episode terminates before HORIZON, undiscounted: the value is the probability of falling within HORIZON actions).
"""
import math
import os
from dataclasses import dataclass

import gymnasium
import mujoco
import numpy as np
from numba import njit
from scipy.special import ndtri

from phase1_reproduction.core.rng import SLOT_START, stream_key, uniform
from phase2_5_qmc.core.onpolicy_select import OnPolicyMoment, episode_stats

XML = os.path.join(os.path.dirname(gymnasium.__file__), "envs", "mujoco", "assets", "hopper.xml")
FRAME_SKIP, DT = 4, 0.008
OBS_DIM, ACT_DIM, N_THETA = 11, 3, 39
HORIZON, GAMMA = 200, 0.99
SLOTS_ACTION = (1, 2, 3)                # random-number slots of the three action dimensions
SLOTS_SHIFT = (7, 8, 9)                 # Array-RQMC's per-step shifts


class Hopper:
    """n Hopper-v5 episodes that share one MuJoCo model and data."""
    SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def __init__(self, n):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.qpos0 = self.model.qpos0.copy()
        self.state = np.empty((n, mujoco.mj_stateSize(self.model, self.SPEC)))

    def _obs(self, out):
        out[:5] = self.data.qpos[1:]
        out[5:] = np.clip(self.data.qvel, -10.0, 10.0)

    def reset(self, k, u, obs):
        """Start episode k from the rest pose plus noise (u: 12 uniforms on [0, 1)); writes its observation."""
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = self.qpos0 - 5e-3 + 1e-2 * u[:6]
        d.qvel[:] = -5e-3 + 1e-2 * u[6:]
        mujoco.mj_forward(self.model, d)
        mujoco.mj_getState(self.model, d, self.state[k], self.SPEC)
        self._obs(obs)

    def step(self, k, a, obs):
        """Apply action a to episode k; writes the next observation and returns (reward, terminated)."""
        d = self.data
        mujoco.mj_setState(self.model, d, self.state[k], self.SPEC)
        x0 = d.qpos[0]
        d.ctrl[:] = a
        mujoco.mj_step(self.model, d, nstep=FRAME_SKIP)
        healthy = (0.7 < d.qpos[1] and -0.2 < d.qpos[2] < 0.2 and np.all(np.abs(d.qpos[2:]) < 100.0)
                   and np.all(np.abs(d.qvel) < 100.0))
        mujoco.mj_getState(self.model, d, self.state[k], self.SPEC)
        self._obs(obs)
        return (d.qpos[0] - x0) / DT + (1.0 if healthy else 0.0) - 1e-3 * float(np.dot(a, a)), not healthy


@dataclass
class Policy:
    """a ~ N(W z + b, diag(exp(log_sd)^2)), z = (s - mean) / std."""
    W: np.ndarray
    b: np.ndarray
    log_sd: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def theta(self):
        return np.concatenate([self.W.ravel(), self.b, self.log_sd])

    def with_theta(self, th):
        return Policy(th[:33].reshape(ACT_DIM, OBS_DIM), th[33:36], th[36:], self.mean, self.std)

    def dist(self, S):
        """Means (m, 3), standard deviations (3,) and normalized states (m, 11) at the states S (m, 11)."""
        Z = (S - self.mean) / self.std
        return Z @ self.W.T + self.b, np.exp(self.log_sd), Z

    @staticmethod
    def log_p(M, sd, A):
        return (-0.5 * ((A - M) / sd) ** 2 - np.log(sd)).sum(-1) - 1.5 * math.log(2 * math.pi)

    @staticmethod
    def score(M, sd, Z, A):
        """grad_theta log pi(a | s) for each row, (m, 39)."""
        r = (A - M) / sd ** 2
        return np.concatenate([(r[:, :, None] * Z[:, None, :]).reshape(len(A), -1), r, ((A - M) / sd) ** 2 - 1], 1)


@njit(cache=True)
def _uniforms(key, seed, js, t, slots, out):
    for i in range(js.shape[0]):
        for d in range(slots.shape[0]):
            out[i, d] = uniform(key, seed, js[i], t, slots[d])


@njit(cache=True)
def _hammersley(m, shifts, out):
    """Rows i = 0 .. m-1: (phi_2(i), phi_3(i), phi_5(i)) + shifts, mod 1 (phi_b = radical inverse in base b)."""
    bases = (2, 3, 5)
    for i in range(m):
        for d in range(3):
            x, f, k = 0.0, 1.0 / bases[d], i
            while k > 0:
                x += f * (k % bases[d])
                k //= bases[d]
                f /= bases[d]
            out[i, d] = (x + shifts[d]) % 1.0


def _reward(r, done, task):
    return (1.0 if done else 0.0) if task == "fall" else r


def _gamma(task):
    return 1.0 if task == "fall" else GAMMA


def _features(Z):
    return np.concatenate([Z, Z * Z, np.ones((len(Z), 1))], 1)


def _start_uniforms(key, seed, j):
    """The 12 uniforms of episode j's start state."""
    return np.array([uniform(key, seed, j, i, SLOT_START) for i in range(12)])


def run(pi_e, q, n, seed, stream, mode="iid", value_w=None, task="return"):
    """One seed: n lockstep episodes from q (None = pi_e). Returns (MC, OIS, PDIS, mean length)."""
    key = stream_key(stream)
    env = Hopper(n)
    S = np.empty((n, OBS_DIM))
    for k in range(n):
        env.reset(k, _start_uniforms(key, seed, k), S[k])
    alive = np.ones(n, bool)
    g, disc, log_w, pdis = np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n)
    js, slots, steps = np.arange(n), np.array(SLOTS_ACTION), 0
    U, shifts = np.empty((n, ACT_DIM)), np.empty(3)
    for t in range(HORIZON):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        steps += idx.size
        Me, sd_e, Z = pi_e.dist(S[idx])
        Mb, sd_b = (Me, sd_e) if q is None else q.dist(S[idx])[:2]
        if mode == "iid":
            _uniforms(key, seed, js[idx], t, slots, U)
            V = U[:idx.size]
        else:                                   # Array-RQMC: rank i (by value) gets the i-th Hammersley point
            for d in range(3):
                shifts[d] = uniform(key, seed, 0, t, SLOTS_SHIFT[d])
            _hammersley(idx.size, shifts, U)
            V = np.empty((idx.size, ACT_DIM))
            V[np.argsort(_features(Z) @ value_w)] = U[:idx.size]
        A = Mb + sd_b * ndtri(np.clip(V, 1e-12, 1 - 1e-12))
        if q is not None:
            log_w[idx] += Policy.log_p(Me, sd_e, A) - Policy.log_p(Mb, sd_b, A)
        for i, k in enumerate(idx):
            r, done = env.step(k, A[i], S[k])
            r = _reward(r, done, task)
            g[k] += disc[k] * r
            pdis[k] += disc[k] * math.exp(log_w[k]) * r
            if done:
                alive[k] = False
        disc[idx] *= _gamma(task)
    return g.mean(), (g * np.exp(log_w)).mean(), pdis.mean(), steps / n


def run_ros(pi_e, lr, n, seed, stream, task="return"):
    """One seed of on-policy ROS over n episodes, one after another. Returns (MC, mean length)."""
    key = stream_key(stream)
    env = Hopper(1)
    th_e = pi_e.theta()
    grad, steps, total = np.zeros(N_THETA), 0, 0.0
    q = pi_e
    s, u = np.empty(OBS_DIM), np.empty((1, ACT_DIM))
    slots = np.array(SLOTS_ACTION)
    for j in range(n):
        env.reset(0, _start_uniforms(key, seed, j), s)
        g, disc = 0.0, 1.0
        for t in range(HORIZON):
            Mb, sd_b, _ = q.dist(s[None])
            _uniforms(key, seed, np.array([j]), t, slots, u)
            a = (Mb + sd_b * ndtri(np.clip(u, 1e-12, 1 - 1e-12)))[0]
            Me, sd_e, Z = pi_e.dist(s[None])
            grad += Policy.score(Me, sd_e, Z, a[None])[0]
            steps += 1
            q = pi_e.with_theta(th_e - lr / steps * grad)
            r, done = env.step(0, a, s)
            g += disc * _reward(r, done, task)
            disc *= _gamma(task)
            if done:
                break
        total += g
    return total / n, steps / n


def train_ars(iters=300, n_dir=8, top=4, step=0.02, noise=0.03, horizon=1000, seed=0):
    """A linear policy a = W (s - mean) / std for Hopper-v5 by Augmented Random Search (V2-t, Mania et al. 2018),
    trained on full 1000-step episodes. Returns (W, mean, std, mean return of the last iteration's rollouts)."""
    rng = np.random.default_rng(seed)
    env = Hopper(1)
    W = np.zeros((ACT_DIM, OBS_DIM))
    count, s_sum, s_sq = 1e-8, np.zeros(OBS_DIM), np.zeros(OBS_DIM)
    o = np.empty(OBS_DIM)

    def rollout(M, mean, std):
        nonlocal count, s_sum, s_sq
        env.reset(0, rng.random(12), o)
        total = 0.0
        for _ in range(horizon):
            count += 1.0                        # running state statistics, used from the next iteration on
            s_sum += o
            s_sq += o * o
            r, done = env.step(0, M @ ((o - mean) / std), o)
            total += r
            if done:
                break
        return total

    returns = np.zeros(1)
    for _ in range(iters):
        mean = s_sum / count
        std = np.sqrt(np.maximum(s_sq / count - mean * mean, 1e-4))
        deltas = rng.standard_normal((n_dir, ACT_DIM, OBS_DIM))
        plus = np.array([rollout(W + noise * d, mean, std) for d in deltas])
        minus = np.array([rollout(W - noise * d, mean, std) for d in deltas])
        best = np.argsort(-np.maximum(plus, minus))[:top]
        sigma_r = np.concatenate([plus[best], minus[best]]).std() + 1e-8
        W += step / (top * sigma_r) * np.einsum("k,kij->ij", plus[best] - minus[best], deltas[best])
        returns = np.concatenate([plus, minus])
    mean = s_sum / count
    return W, mean, np.sqrt(np.maximum(s_sq / count - mean * mean, 1e-4)), returns.mean()


def rollout(pi, j, seed, stream, env=None, task="return"):
    """Episode j of policy pi with the counter-based uniforms: states (T, 11), actions (T, 3), rewards (T,)."""
    key = stream_key(stream)
    env = env or Hopper(1)
    s, u, slots = np.empty(OBS_DIM), np.empty((1, ACT_DIM)), np.array(SLOTS_ACTION)
    env.reset(0, _start_uniforms(key, seed, j), s)
    S, A, R = [], [], []
    for t in range(HORIZON):
        M, sd, _ = pi.dist(s[None])
        _uniforms(key, seed, np.array([j]), t, slots, u)
        a = (M + sd * ndtri(np.clip(u, 1e-12, 1 - 1e-12)))[0]
        S.append(s.copy())
        A.append(a)
        r, done = env.step(0, a, s)
        R.append(_reward(r, done, task))
        if done:
            break
    return np.array(S), np.array(A), np.array(R)


def _returns_to_go(R, gamma):
    G = np.empty(len(R))
    acc = 0.0
    for t in range(len(R) - 1, -1, -1):
        acc = R[t] + gamma * acc
        G[t] = acc
    return G


def _value_moments(args):
    pi_e, js, stream, task = args
    env, xx, xy = Hopper(1), np.zeros((23, 23)), np.zeros(23)
    for j in js:
        S, _, R = rollout(pi_e, j, 0, stream, env, task)
        F = _features((S - pi_e.mean) / pi_e.std)
        xx += F.T @ F
        xy += F.T @ _returns_to_go(R, _gamma(task))
    return xx, xy


def fit_value(pi_e, n=4000, stream="value-fit", map_fn=map, ridge=1e-6, task="return"):
    """Sort key: least-squares fit of pi_e's discounted return to go on [z, z^2, 1], from n on-policy episodes."""
    parts = list(map_fn(_value_moments, [(pi_e, range(c, n, 40), stream, task) for c in range(40)]))
    A, b = sum(p[0] for p in parts), sum(p[1] for p in parts)
    return np.linalg.solve(A + ridge * np.trace(A) * np.eye(23), b)


def _bpg_part(args):
    """Episodes js from q: (sum of IS^2 * sum_t grad log q(a_t|s_t), the episodes' IS estimates)."""
    pi_e, q, js, seed, stream, task = args
    env, grad, w = Hopper(1), np.zeros(N_THETA), []
    for j in js:
        S, A, R = rollout(q, j, seed, stream, env, task)
        Mb, sd_b, Z = q.dist(S)
        Me, sd_e, _ = pi_e.dist(S)
        disc = _gamma(task) ** np.arange(len(R))
        is_ = (disc * R).sum() * math.exp((Policy.log_p(Me, sd_e, A) - Policy.log_p(Mb, sd_b, A)).sum())
        grad += is_ * is_ * Policy.score(Mb, sd_b, Z, A).sum(0)
        w.append(is_)
    return grad, w


def _onpolicy_part(args):
    """On-policy episodes js of pi_e (seed 0) with a nonzero return, as onpolicy_select.episode_stats."""
    pi_e, js, stream, task = args
    env, out = Hopper(1), []
    for j in js:
        S, A, R = rollout(pi_e, j, 0, stream, env, task)
        g = float((_gamma(task) ** np.arange(len(R)) * R).sum())
        if g != 0.0:
            out.append(episode_stats(pi_e, S, A, g))
    return out


def onpolicy_moment(pi_e, n, stream, map_fn=map, task="return"):
    """q -> an unbiased estimate of q's OIS second moment from n fixed on-policy episodes (onpolicy_select)."""
    parts = map_fn(_onpolicy_part, [(pi_e, range(c, n, 40), stream, task) for c in range(40)])
    return OnPolicyMoment(pi_e, [x for p in parts for x in p], n)


def bpg_target(pi_e, iters=100, batch=1000, step=0.01, n_val=4000, every=10, stream="bpg", map_fn=map,
               task="return", select="onpolicy", n_on=40000, trace=None):
    """A fixed BPS target, learned offline: from pi_e, unit-norm steps that lower the OIS second moment E_q[(g w)^2].
    Every `every` steps the iterate's second moment is measured and the best iterate (pi_e included) is returned:
    select="onpolicy" measures it on one fixed sample of n_on episodes of pi_e (onpolicy_select); select="behavior"
    (the rule of the first tables) on n_val episodes of the iterate itself, which a q that reaches a rare event by
    routes pi_e does not take can fool. A list `trace` receives (step, second moment) at every check."""
    if select == "onpolicy":
        second_moment = onpolicy_moment(pi_e, n_on, stream + "-onpolicy", map_fn, task)
    else:
        def second_moment(q):
            parts = map_fn(_bpg_part, [(pi_e, q, range(c, n_val, 40), 0, stream + "-val", task) for c in range(40)])
            return np.mean([x * x for _, w in parts for x in w])

    th = pi_e.theta()
    best = (second_moment(pi_e), th.copy())
    if trace is not None:
        trace.append((0, best[0]))
    for k in range(iters):
        q = pi_e.with_theta(th)
        parts = list(map_fn(_bpg_part, [(pi_e, q, range(c, batch, 40), k + 1, stream, task) for c in range(40)]))
        direction = sum(p[0] for p in parts)            # = batch * (minus the gradient of the second moment)
        th = th + step / math.sqrt(1.0 + k / 50) * direction / np.linalg.norm(direction)
        if (k + 1) % every == 0:
            m2 = second_moment(pi_e.with_theta(th))
            if trace is not None:
                trace.append((k + 1, m2))
            if m2 < best[0]:
                best = (m2, th.copy())
    return pi_e.with_theta(best[1])
