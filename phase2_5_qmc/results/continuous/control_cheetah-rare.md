# Coordinated BPS in continuous control: HalfCheetah, probability that the torso turns past 1.57 rad within 200 actions

pi_e: ARS linear policy with action noise sd 0.05; dynamics noise sd 0.02 on the applied control. Truth v = 0.014978 ± 0.0002 (400000 episodes), Var(g) = 0.01475, mean length 199.0. BPS target learned offline by BPG in 184 s (not counted; iterates compared on 40000 episodes of pi_e). No BPG iterate beat pi_e on validation, so q = pi_e and the BPS rows equal the on-policy rows. n·MSE/Var(g): on-policy MC = 1; 200 seeds. Coverage: 95% t-intervals from groups of 10 runs. Time: µs per episode at n = 2048 on 10 processes. ROS grid: 3e-05, 0.0001, 0.0003, 0.001.

| method | n = 32 | n = 256 | n = 2048 | equal steps, n = 2048 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.97 | 0.98 | 0.961 | 0.961 | -2.4 | 95% | 1.00 | 829 | – |
| ROS-MC | 0.732 | 0.938 | 0.896 | 0.896 | -1.1 | 85% | 0.96 | 1338 | 0.0003, 0.0003, 0.0001 |
| BPS + OIS (i.i.d.) | 0.97 | 0.98 | 0.961 | 0.961 | -2.4 | 95% | 1.00 | 761 | – |
| BPS + PDIS (i.i.d.) | 0.97 | 0.98 | 0.961 | 0.961 | -2.4 | 95% | 1.00 | 761 | – |
| Array-RQMC, on-policy (MC) | 0.887 | 1.11 | 0.959 | 0.959 | -1.7 | 80% | 0.99 | 794 | – |
| BPS + Array-RQMC, OIS | 0.887 | 1.11 | 0.959 | 0.959 | -1.7 | 80% | 0.99 | 779 | – |
| BPS + Array-RQMC, PDIS | 0.887 | 1.11 | 0.959 | 0.959 | -1.7 | 80% | 0.99 | 779 | – |

## Paired MSE ratios per episode

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| comparison | n = 32 | n = 256 | n = 2048 |
|---|---|---|---|
| BPS + Array-RQMC, OIS vs BPS + OIS (i.i.d.) | 0.915 ~ | 1.13 ~ | 0.998 ~ |
| BPS + Array-RQMC, OIS vs Array-RQMC, on-policy (MC) | 1 ~ | 1 ~ | 1 ~ |
| BPS + Array-RQMC, OIS vs ROS-MC | 1.21 ~ | 1.18 ~ | 1.07 ~ |
| BPS + Array-RQMC, PDIS vs ROS-MC | 1.21 ~ | 1.18 ~ | 1.07 ~ |
| BPS + OIS (i.i.d.) vs OS-MC | 1 ~ | 1 ~ | 1 ~ |
| Array-RQMC, on-policy (MC) vs OS-MC | 0.915 ~ | 1.13 ~ | 0.998 ~ |
