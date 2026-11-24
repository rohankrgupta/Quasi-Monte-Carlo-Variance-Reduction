# Coordinated BPS in continuous control: Hopper, discounted return over 200 actions

pi_e: ARS linear policy (mean return 1101 on full 1000-step episodes) with action noise sd 0.2. Truth v = 88.07 ± 0.0007 (400000 episodes), Var(g) = 0.1973, mean length 200.0. BPS target learned offline by BPG in 329 s (not counted). No BPG iterate beat pi_e on validation, so q = pi_e and the BPS rows equal the on-policy rows. n·MSE/Var(g): on-policy MC = 1; 200 seeds. Coverage: 95% t-intervals from groups of 10 runs. Time: µs per episode at n = 2048 on 10 processes.

| method | n = 32 | n = 256 | n = 2048 | equal steps, n = 2048 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.907 | 0.938 | 1.01 | 1.01 | -1.1 | 90% | 1.00 | 1923 | – |
| ROS-MC | 0.72 | 0.942 | 0.916 | 0.916 | -1.4 | 100% | 0.96 | 2332 | 0.003, 0.0001, 0.003 |
| BPS + OIS (i.i.d.) | 0.907 | 0.938 | 1.01 | 1.01 | -1.1 | 90% | 1.00 | 1877 | – |
| BPS + PDIS (i.i.d.) | 0.907 | 0.938 | 1.01 | 1.01 | -1.1 | 90% | 1.00 | 1877 | – |
| Array-RQMC, on-policy (MC) | 0.781 | 0.644 | 0.734 | 0.734 | -0.6 | 100% | 0.84 | 1929 | – |
| BPS + Array-RQMC, OIS | 0.781 | 0.644 | 0.734 | 0.734 | -0.6 | 100% | 0.84 | 2006 | – |
| BPS + Array-RQMC, PDIS | 0.781 | 0.644 | 0.734 | 0.734 | -0.6 | 100% | 0.84 | 2006 | – |

## Paired MSE ratios per episode

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| comparison | n = 32 | n = 256 | n = 2048 |
|---|---|---|---|
| BPS + Array-RQMC, OIS vs BPS + OIS (i.i.d.) | 0.861 ~ | 0.686 ✓ | 0.727 ✓ |
| BPS + Array-RQMC, PDIS vs BPS + PDIS (i.i.d.) | 0.861 ~ | 0.686 ✓ | 0.727 ✓ |
| BPS + Array-RQMC, OIS vs ROS-MC | 1.09 ~ | 0.684 ✓ | 0.801 ~ |
| BPS + Array-RQMC, PDIS vs ROS-MC | 1.09 ~ | 0.684 ✓ | 0.801 ~ |
| Array-RQMC, on-policy (MC) vs OS-MC | 0.861 ~ | 0.686 ✓ | 0.727 ✓ |
| Array-RQMC, on-policy (MC) vs ROS-MC | 1.09 ~ | 0.684 ✓ | 0.801 ~ |
