"""Continuous-control domains (gymnasium's Reacher-v5 and HalfCheetah-v5) for coordinated BPS, with a linear Gaussian
evaluation policy. Generalizes core/hopper.py to any action dimension and adds stochastic dynamics.

Environments: gymnasium's model files and rules, stepped with MuJoCo directly so that many episodes can run in
lockstep (with the same start state and actions each reproduces gymnasium.make(...) exactly). All episodes share one
MjData; each episode's integration state is swapped in before its step.
Dynamics noise: the applied control is a + env_sd * N(0, I), drawn from a separate random stream that no sampler
controls; the reward's control cost uses a itself.

Policy: a ~ N(W z + b, diag(sd^2)), z = (s - mean) / std, W trained by ARS (train_ars); theta = (W, b, log sd).
Every action is mean + sd * Phi^{-1}(u), so samplers differ only in how they choose the uniforms:
  * iid:   counter-based uniforms (rng.uniform at (seed, episode, step, dimension)).
  * array: Array-RQMC. At each step the running episodes are sorted by a value estimate V(s) = w . [z, z^2, 1], and
           the episode of rank i gets the i-th Halton point in the action dimensions + a per-step shift, mod 1.
  * ros:   ROS as in the ROS repo, episodes one after another: before each action the behavior is
           theta_e - lr * (mean over all past steps of grad log pi_e(a|s)).
Tasks, on the same policy:
  * "return": the domain's discounted return (gamma = 0.99) over its horizon;
  * "rare":   the probability of a failure, undiscounted (Reacher: the fingertip ends more than `threshold` from the
              target; HalfCheetah: the torso turns more than `threshold` radians, which ends the episode).
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

ASSETS = os.path.join(os.path.dirname(gymnasium.__file__), "envs", "mujoco", "assets")
GAMMA = 0.99
PRIMES = (2, 3, 5, 7, 11, 13)
SLOT_SHIFT0 = 7                          # Array-RQMC's per-step shifts use slots 7 .. 7 + act_dim - 1


class Domain:
    """n episodes of one MuJoCo task that share one model and data. Subclasses set the task's constants and rules."""
    SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
    xml = frame_skip = obs_dim = act_dim = horizon = n_start = None
    rare_at_end = True                   # the rare event is checked at the last step (else at every step, and ends it)

    def __init__(self, n, env_sd=0.0, mass=1.0, friction=1.0):
        """mass scales every body's mass and inertia; friction scales every geom's sliding friction."""
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, self.xml))
        self.model.body_mass[:] *= mass
        self.model.body_inertia[:] *= mass
        self.model.geom_friction[:, 0] *= friction
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * self.frame_skip
        self.state = np.empty((n, mujoco.mj_stateSize(self.model, self.SPEC)))
        self.env_sd = env_sd

    def reset(self, k, u, obs):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        self._start(u)
        mujoco.mj_forward(self.model, d)
        mujoco.mj_getState(self.model, d, self.state[k], self.SPEC)
        self._obs(obs)

    def step(self, k, a, noise, obs):
        """Apply action a (+ env_sd * noise) to episode k; writes the next observation and returns
        (reward, terminated, the rare-event statistic)."""
        d = self.data
        mujoco.mj_setState(self.model, d, self.state[k], self.SPEC)
        x0 = d.qpos[0]
        d.ctrl[:] = a + self.env_sd * noise
        mujoco.mj_step(self.model, d, nstep=self.frame_skip)
        mujoco.mj_getState(self.model, d, self.state[k], self.SPEC)
        self._obs(obs)
        return self._reward(a, x0)


class Reacher(Domain):
    """Reacher-v5: a 2-link arm reaching a random target; reward = -distance - |a|^2, 50 steps, no termination."""
    xml, frame_skip, obs_dim, act_dim, horizon, n_start = "reacher.xml", 2, 10, 2, 50, 10

    def _start(self, u):
        d = self.data
        d.qpos[:] = self.model.qpos0 - 0.1 + 0.2 * u[:4]
        r, phi = 0.2 * math.sqrt(u[4]), 2 * math.pi * u[5]          # uniform on the disc of radius 0.2
        d.qpos[2:] = r * math.cos(phi), r * math.sin(phi)
        d.qvel[:] = -0.005 + 0.01 * u[6:10]
        d.qvel[2:] = 0.0

    def _tip(self):
        return self.data.body("fingertip").xpos - self.data.body("target").xpos

    def _obs(self, out):
        q = self.data.qpos
        out[:2], out[2:4], out[4:6] = np.cos(q[:2]), np.sin(q[:2]), q[2:]
        out[6:8], out[8:10] = self.data.qvel[:2], self._tip()[:2]

    def _reward(self, a, x0):
        dist = float(np.linalg.norm(self._tip()))
        return -dist - float(np.dot(a, a)), False, dist


class HalfCheetah(Domain):
    """HalfCheetah-v5: reward = forward velocity - 0.1 |a|^2, no termination; truncated at 200 actions."""
    xml, frame_skip, obs_dim, act_dim, horizon, n_start = "half_cheetah.xml", 5, 17, 6, 200, 18
    rare_at_end = False

    def _start(self, u):
        d = self.data
        d.qpos[:] = self.model.qpos0 - 0.1 + 0.2 * u[:9]
        d.qvel[:] = 0.1 * ndtri(np.clip(u[9:18], 1e-12, 1 - 1e-12))

    def _obs(self, out):
        out[:8], out[8:] = self.data.qpos[1:], self.data.qvel

    def _reward(self, a, x0):
        return (self.data.qpos[0] - x0) / self.dt - 0.1 * float(np.dot(a, a)), False, abs(float(self.data.qpos[2]))


DOMAINS = {"reacher": Reacher, "cheetah": HalfCheetah}


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
        k, m = self.W.shape
        return Policy(th[:k * m].reshape(k, m), th[k * m:k * m + k], th[k * m + k:], self.mean, self.std)

    def dist(self, S):
        Z = (S - self.mean) / self.std
        return Z @ self.W.T + self.b, np.exp(self.log_sd), Z

    @staticmethod
    def log_p(M, sd, A):
        return (-0.5 * ((A - M) / sd) ** 2 - np.log(sd)).sum(-1) - 0.5 * A.shape[-1] * math.log(2 * math.pi)

    @staticmethod
    def score(M, sd, Z, A):
        """grad_theta log pi(a | s) for each row."""
        r = (A - M) / sd ** 2
        return np.concatenate([(r[:, :, None] * Z[:, None, :]).reshape(len(A), -1), r, ((A - M) / sd) ** 2 - 1], 1)


@dataclass
class Task:
    """What is estimated: the domain, the dynamics noise, "return" or "rare" (with its threshold), and the physics
    (mass and friction scales; 1 = the simulator's nominal model)."""
    domain: str
    env_sd: float = 0.0
    kind: str = "return"
    threshold: float = 0.0
    mass: float = 1.0
    friction: float = 1.0

    def env(self, n):
        return DOMAINS[self.domain](n, self.env_sd, self.mass, self.friction)

    @property
    def gamma(self):
        return GAMMA if self.kind == "return" else 1.0

    def reward(self, env, r, x, t):
        """(task reward, whether the task ends the episode) after a step at time t."""
        if self.kind == "return":
            return r, False
        if env.rare_at_end:
            return (1.0 if t == env.horizon - 1 and x > self.threshold else 0.0), False
        return (1.0, True) if x > self.threshold else (0.0, False)


@njit(cache=True)
def _uniforms(key, seed, js, t, d, out):
    for i in range(js.shape[0]):
        for c in range(d):
            out[i, c] = uniform(key, seed, js[i], t, 1 + c)


@njit(cache=True)
def _halton(m, d, shifts, out):
    """Rows i = 0 .. m-1: (phi_2(i), phi_3(i), ...) in d dimensions + shifts, mod 1."""
    bases = (2, 3, 5, 7, 11, 13)
    for i in range(m):
        for c in range(d):
            x, f, k = 0.0, 1.0 / bases[c], i
            while k > 0:
                x += f * (k % bases[c])
                k //= bases[c]
                f /= bases[c]
            out[i, c] = (x + shifts[c]) % 1.0


def _start_uniforms(env, key, seed, j):
    return np.array([uniform(key, seed, j, i, SLOT_START) for i in range(env.n_start)])


def _features(Z):
    return np.concatenate([Z, Z * Z, np.ones((len(Z), 1))], 1)


def _gauss(U):
    return ndtri(np.clip(U, 1e-12, 1 - 1e-12))


def run(task, pi_e, q, n, seed, stream, mode="iid", value_w=None):
    """One seed: n lockstep episodes from q (None = pi_e). Returns (MC, OIS, PDIS, mean length)."""
    key, key_env = stream_key(stream), stream_key(stream + "/dynamics")
    env = task.env(n)
    d, S = env.act_dim, np.empty((n, env.obs_dim))
    for k in range(n):
        env.reset(k, _start_uniforms(env, key, seed, k), S[k])
    alive = np.ones(n, bool)
    g, disc, log_w, pdis = np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n)
    js, steps = np.arange(n), 0
    U, E, shifts = np.empty((n, d)), np.empty((n, d)), np.empty(d)
    for t in range(env.horizon):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        steps += idx.size
        Me, sd_e, Z = pi_e.dist(S[idx])
        Mb, sd_b = (Me, sd_e) if q is None else q.dist(S[idx])[:2]
        if mode == "iid":
            _uniforms(key, seed, js[idx], t, d, U)
            V = U[:idx.size]
        else:                                   # Array-RQMC: rank i (by value) gets the i-th Halton point
            for c in range(d):
                shifts[c] = uniform(key, seed, 0, t, SLOT_SHIFT0 + c)
            _halton(idx.size, d, shifts, U)
            V = np.empty((idx.size, d))
            V[np.argsort(_features(Z) @ value_w)] = U[:idx.size]
        A = Mb + sd_b * _gauss(V)
        if q is not None:
            log_w[idx] += Policy.log_p(Me, sd_e, A) - Policy.log_p(Mb, sd_b, A)
        _uniforms(key_env, seed, js[idx], t, d, E)
        noise = _gauss(E[:idx.size])
        for i, k in enumerate(idx):
            r, _, x = env.step(k, A[i], noise[i], S[k])
            r, done = task.reward(env, r, x, t)
            g[k] += disc[k] * r
            pdis[k] += disc[k] * math.exp(log_w[k]) * r
            if done:
                alive[k] = False
        disc[idx] *= task.gamma
    return g.mean(), (g * np.exp(log_w)).mean(), pdis.mean(), steps / n


def run_ros(task, pi_e, lr, n, seed, stream):
    """One seed of on-policy ROS over n episodes, one after another. Returns (MC, mean length)."""
    key, key_env = stream_key(stream), stream_key(stream + "/dynamics")
    env = task.env(1)
    d, th_e = env.act_dim, pi_e.theta()
    grad, steps, total, q = np.zeros(th_e.size), 0, 0.0, pi_e
    s, u, e = np.empty(env.obs_dim), np.empty((1, d)), np.empty((1, d))
    for j in range(n):
        env.reset(0, _start_uniforms(env, key, seed, j), s)
        g, disc = 0.0, 1.0
        for t in range(env.horizon):
            Mb, sd_b, _ = q.dist(s[None])
            _uniforms(key, seed, np.array([j]), t, d, u)
            a = (Mb + sd_b * _gauss(u))[0]
            Me, sd_e, Z = pi_e.dist(s[None])
            grad += Policy.score(Me, sd_e, Z, a[None])[0]
            steps += 1
            q = pi_e.with_theta(th_e - lr / steps * grad)
            _uniforms(key_env, seed, np.array([j]), t, d, e)
            r, _, x = env.step(0, a, _gauss(e)[0], s)
            r, done = task.reward(env, r, x, t)
            g += disc * r
            disc *= task.gamma
            if done:
                break
        total += g
    return total / n, steps / n


def rollout(task, pi, j, seed, stream, env=None):
    """Episode j of policy pi: states (T, obs_dim), actions (T, act_dim), task rewards (T,)."""
    key, key_env = stream_key(stream), stream_key(stream + "/dynamics")
    env = env or task.env(1)
    d = env.act_dim
    s, u, e = np.empty(env.obs_dim), np.empty((1, d)), np.empty((1, d))
    env.reset(0, _start_uniforms(env, key, seed, j), s)
    S, A, R = [], [], []
    for t in range(env.horizon):
        M, sd, _ = pi.dist(s[None])
        _uniforms(key, seed, np.array([j]), t, d, u)
        a = (M + sd * _gauss(u))[0]
        S.append(s.copy())
        A.append(a)
        _uniforms(key_env, seed, np.array([j]), t, d, e)
        r, _, x = env.step(0, a, _gauss(e)[0], s)
        r, done = task.reward(env, r, x, t)
        R.append(r)
        if done:
            break
    return np.array(S), np.array(A), np.array(R)


def _returns_to_go(R, gamma):
    G, acc = np.empty(len(R)), 0.0
    for t in range(len(R) - 1, -1, -1):
        acc = R[t] + gamma * acc
        G[t] = acc
    return G


def _value_moments(args):
    task, pi_e, js, stream = args
    env = task.env(1)
    m = 2 * env.obs_dim + 1
    xx, xy = np.zeros((m, m)), np.zeros(m)
    for j in js:
        S, _, R = rollout(task, pi_e, j, 0, stream, env)
        F = _features((S - pi_e.mean) / pi_e.std)
        xx += F.T @ F
        xy += F.T @ _returns_to_go(R, task.gamma)
    return xx, xy


def fit_value(task, pi_e, n=4000, stream="value-fit", map_fn=map, ridge=1e-6):
    """Sort key: least-squares fit of pi_e's return to go on [z, z^2, 1], from n on-policy episodes."""
    parts = list(map_fn(_value_moments, [(task, pi_e, range(c, n, 40), stream) for c in range(40)]))
    A, b = sum(p[0] for p in parts), sum(p[1] for p in parts)
    return np.linalg.solve(A + ridge * np.trace(A) * np.eye(len(b)), b)


def _bpg_part(args):
    """Episodes js from q: (sum of IS^2 * sum_t grad log q(a_t|s_t), the episodes' IS estimates)."""
    task, pi_e, q, js, seed, stream = args
    env, grad, w = task.env(1), np.zeros(pi_e.theta().size), []
    for j in js:
        S, A, R = rollout(task, q, j, seed, stream, env)
        Mb, sd_b, Z = q.dist(S)
        Me, sd_e, _ = pi_e.dist(S)
        disc = task.gamma ** np.arange(len(R))
        is_ = (disc * R).sum() * math.exp((Policy.log_p(Me, sd_e, A) - Policy.log_p(Mb, sd_b, A)).sum())
        grad += is_ * is_ * Policy.score(Mb, sd_b, Z, A).sum(0)
        w.append(is_)
    return grad, w


def _onpolicy_part(args):
    """On-policy episodes js of pi_e (seed 0) with a nonzero return, as onpolicy_select.episode_stats."""
    task, pi_e, js, stream = args
    env, out = task.env(1), []
    for j in js:
        S, A, R = rollout(task, pi_e, j, 0, stream, env)
        g = float((task.gamma ** np.arange(len(R)) * R).sum())
        if g != 0.0:
            out.append(episode_stats(pi_e, S, A, g))
    return out


def onpolicy_moment(task, pi_e, n, stream, map_fn=map):
    """q -> an unbiased estimate of q's OIS second moment from n fixed on-policy episodes (onpolicy_select)."""
    parts = map_fn(_onpolicy_part, [(task, pi_e, range(c, n, 40), stream) for c in range(40)])
    return OnPolicyMoment(pi_e, [x for p in parts for x in p], n)


def bpg_target(task, pi_e, iters=100, batch=1000, step=0.01, n_val=4000, every=10, stream="bpg", map_fn=map,
               select="onpolicy", n_on=40000, trace=None):
    """A fixed BPS target, learned offline: from pi_e, unit-norm steps that lower the OIS second moment E_q[(g w)^2].
    Every `every` steps the iterate's second moment is measured and the best iterate (pi_e included) is returned:
    select="onpolicy" measures it on one fixed sample of n_on episodes of pi_e (onpolicy_select); select="behavior"
    (the rule of the first tables) on n_val episodes of the iterate itself, which a q that reaches a rare event by
    routes pi_e does not take can fool. A list `trace` receives (step, second moment) at every check."""
    if select == "onpolicy":
        second_moment = onpolicy_moment(task, pi_e, n_on, stream + "-onpolicy", map_fn)
    else:
        def second_moment(q):
            parts = map_fn(_bpg_part, [(task, pi_e, q, range(c, n_val, 40), 0, stream + "-val") for c in range(40)])
            return np.mean([x * x for _, w in parts for x in w])

    th = pi_e.theta()
    best = (second_moment(pi_e), th.copy())
    if trace is not None:
        trace.append((0, best[0]))
    for k in range(iters):
        q = pi_e.with_theta(th)
        parts = list(map_fn(_bpg_part, [(task, pi_e, q, range(c, batch, 40), k + 1, stream) for c in range(40)]))
        direction = sum(p[0] for p in parts)
        th = th + step / math.sqrt(1.0 + k / 50) * direction / np.linalg.norm(direction)
        if (k + 1) % every == 0:
            m2 = second_moment(pi_e.with_theta(th))
            if trace is not None:
                trace.append((k + 1, m2))
            if m2 < best[0]:
                best = (m2, th.copy())
    return pi_e.with_theta(best[1])


def train_ars(domain, iters=300, n_dir=8, top=4, step=0.02, noise=0.03, horizon=None, seed=0):
    """A linear policy a = W (s - mean) / std by Augmented Random Search (V2-t, Mania et al. 2018), noise-free
    dynamics. Returns (W, mean, std, mean return of the last iteration's rollouts)."""
    rng = np.random.default_rng(seed)
    env = DOMAINS[domain](1)
    horizon = horizon or env.horizon
    W = np.zeros((env.act_dim, env.obs_dim))
    count, s_sum, s_sq = 1e-8, np.zeros(env.obs_dim), np.zeros(env.obs_dim)
    o, zero = np.empty(env.obs_dim), np.zeros(env.act_dim)

    def episode(M, mean, std):
        nonlocal count, s_sum, s_sq
        env.reset(0, rng.random(env.n_start), o)
        total = 0.0
        for _ in range(horizon):
            count += 1.0
            s_sum += o
            s_sq += o * o
            r, done, _ = env.step(0, M @ ((o - mean) / std), zero, o)
            total += r
            if done:
                break
        return total

    returns = np.zeros(1)
    for _ in range(iters):
        mean = s_sum / count
        std = np.sqrt(np.maximum(s_sq / count - mean * mean, 1e-4))
        deltas = rng.standard_normal((n_dir, env.act_dim, env.obs_dim))
        plus = np.array([episode(W + noise * dl, mean, std) for dl in deltas])
        minus = np.array([episode(W - noise * dl, mean, std) for dl in deltas])
        best = np.argsort(-np.maximum(plus, minus))[:top]
        sigma_r = np.concatenate([plus[best], minus[best]]).std() + 1e-8
        W += step / (top * sigma_r) * np.einsum("k,kij->ij", plus[best] - minus[best], deltas[best])
        returns = np.concatenate([plus, minus])
    mean = s_sum / count
    return W, mean, np.sqrt(np.maximum(s_sq / count - mean * mean, 1e-4)), returns.mean()
