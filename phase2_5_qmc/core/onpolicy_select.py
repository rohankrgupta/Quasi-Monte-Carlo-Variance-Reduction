"""Choosing among BPG's iterates by their OIS second moment measured on pi_e's own episodes.

For a behavior q, the second moment of the per-episode OIS estimate is E_q[(g w)^2] = E_pi_e[g^2 pi_e(tau) / q(tau)]
(w = pi_e(tau) / q(tau)), so one fixed sample of N on-policy episodes estimates it without bias for every candidate q,
and compares candidates on common episodes. Measured on q's own episodes instead, the same quantity can be fooled by a
q that reaches a rare event by routes pi_e does not take: q's sample then misses the episodes on which pi_e reaches
the event and q is unlikely, the ones with the largest weights. On pi_e's episodes those enter with weight
pi_e / q >> 1.

For the linear-Gaussian policies of the MuJoCo domains (a ~ N(W z + b, diag(sd^2)), z = (s - mean) / std, with pi_e's
normalization for every q), log pi_e(tau) / q(tau) depends on an episode only through a few sums, so each episode is
stored as those sums: with x_t = [z_t, 1] and r_t = a_t - mean_e(s_t),
    sum_t x_t x_t^T,  sum_t r_t x_t^T,  sum_t r_t^2 (per action dimension),  T,  g.
Episodes with g = 0 contribute nothing and are not stored.
"""
import numpy as np


def episode_stats(pi_e, S, A, g):
    """The sums above for one on-policy episode (states S, actions A, return g)."""
    Me, _, Z = pi_e.dist(S)
    X = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    R = A - Me
    return X.T @ X, R.T @ X, (R * R).sum(0), len(S), g


class OnPolicyMoment:
    """q -> (1/n) sum over n on-policy episodes of g^2 pi_e(tau) / q(tau), from the stats of those with g != 0."""

    def __init__(self, pi_e, stats, n):
        self.n, self.log_sd_e = n, np.asarray(pi_e.log_sd, float)
        self.th_e = np.concatenate([pi_e.W, np.asarray(pi_e.b, float)[:, None]], 1)
        self.empty = len(stats) == 0
        if not self.empty:
            self.XX = np.stack([s[0] for s in stats])
            self.RX = np.stack([s[1] for s in stats])
            self.RR = np.stack([s[2] for s in stats])
            self.T = np.array([s[3] for s in stats], float)
            self.g = np.array([s[4] for s in stats], float)
        self.mean_g = 0.0 if self.empty else self.g.sum() / n

    def log_ratio(self, q):
        """log pi_e(tau) / q(tau) for every stored episode."""
        D = np.concatenate([q.W, np.asarray(q.b, float)[:, None]], 1) - self.th_e     # mean_q - mean_e = D x
        quad = np.einsum("cm,emn,cn->ec", D, self.XX, D, optimize=True)
        cross = np.einsum("ecm,cm->ec", self.RX, D)
        iv_e, iv_q = np.exp(-2 * self.log_sd_e), np.exp(-2 * np.asarray(q.log_sd, float))
        return ((-0.5 * self.RR * iv_e + 0.5 * (self.RR - 2 * cross + quad) * iv_q).sum(1)
                + self.T * (np.asarray(q.log_sd, float) - self.log_sd_e).sum())

    def __call__(self, q):
        if self.empty:
            return 0.0
        with np.errstate(over="ignore"):
            return float((self.g ** 2 * np.exp(self.log_ratio(q))).sum() / self.n)
