# Coordinated BPS in continuous control: HalfCheetah, discounted return over 200 actions

pi_e: ARS linear policy with action noise sd 0.05; dynamics noise sd 0.02 on the applied control. Truth v = 232.37 ± 0.043 (400000 episodes), Var(g) = 724.6, mean length 200.0. BPS target learned offline by BPG in 145 s (not counted). No BPG iterate beat pi_e on validation, so q = pi_e and the BPS rows equal the on-policy rows. n·MSE/Var(g): on-policy MC = 1; 200 seeds. Coverage: 95% t-intervals from groups of 10 runs. Time: µs per episode at n = 2048 on 10 processes. ROS grid: 3e-05, 0.0001, 0.0003, 0.001.

| method | n = 32 | n = 256 | n = 2048 | equal steps, n = 2048 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.868 | 1.03 | 1.16 | 1.16 | +1.7 | 100% | 1.00 | 718 | – |
| ROS-MC | 0.938 | 1.1 | 1.21 | 1.21 | +1.3 | 95% | 1.01 | 1253 | 0.0003, 3e-05, 3e-05 |
| BPS + OIS (i.i.d.) | 0.868 | 1.03 | 1.16 | 1.16 | +1.7 | 100% | 1.00 | 717 | – |
| BPS + PDIS (i.i.d.) | 0.868 | 1.03 | 1.16 | 1.16 | +1.7 | 100% | 1.00 | 717 | – |
| Array-RQMC, on-policy (MC) | 0.977 | 1.07 | 0.908 | 0.908 | +0.4 | 100% | 0.88 | 718 | – |
| BPS + Array-RQMC, OIS | 0.977 | 1.07 | 0.908 | 0.908 | +0.4 | 100% | 0.88 | 720 | – |
| BPS + Array-RQMC, PDIS | 0.977 | 1.07 | 0.908 | 0.908 | +0.4 | 100% | 0.88 | 720 | – |

## Paired MSE ratios per episode

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| comparison | n = 32 | n = 256 | n = 2048 |
|---|---|---|---|
| BPS + Array-RQMC, OIS vs BPS + OIS (i.i.d.) | 1.13 ~ | 1.04 ~ | 0.785 ~ |
| BPS + Array-RQMC, OIS vs Array-RQMC, on-policy (MC) | 1 ~ | 1 ~ | 1 ~ |
| BPS + Array-RQMC, OIS vs ROS-MC | 1.04 ~ | 0.969 ~ | 0.749 ✓ |
| BPS + Array-RQMC, PDIS vs ROS-MC | 1.04 ~ | 0.969 ~ | 0.749 ✓ |
| BPS + OIS (i.i.d.) vs OS-MC | 1 ~ | 1 ~ | 1 ~ |
| Array-RQMC, on-policy (MC) vs OS-MC | 1.13 ~ | 1.04 ~ | 0.785 ~ |
