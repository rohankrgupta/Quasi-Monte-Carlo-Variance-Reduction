# Coordinated BPS in continuous control: Hopper, probability of falling within 200 actions

pi_e: ARS linear policy (mean return 1101 on full 1000-step episodes) with action noise sd 0.75. Truth v = 0.0073975 ± 0.00013 (400000 episodes), Var(g) = 0.007343, mean length 198.8. BPS target learned offline by BPG in 325 s (not counted; iterates compared on 40000 episodes of pi_e). OIS = PDIS: the only reward is at an episode's last step. n·MSE/Var(g): on-policy MC = 1; 200 seeds. Coverage: 95% t-intervals from groups of 10 runs. Time: µs per episode at n = 2048 on 10 processes.

| method | n = 32 | n = 256 | n = 2048 | equal steps, n = 2048 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.884 | 1.11 | 0.979 | 0.979 | +0.8 | 95% | 1.00 | 3247 | – |
| ROS-MC | 0.935 | 1.02 | 0.967 | 0.967 | +0.5 | 95% | 1.01 | 4178 | 0.0001, 0.0001, 0.0001 |
| BPS + OIS (i.i.d.) | 0.571 | 0.631 | 0.537 | 0.527 | +0.3 | 100% | 0.74 | 3243 | – |
| BPS + PDIS (i.i.d.) | 0.571 | 0.631 | 0.537 | 0.527 | +0.3 | 100% | 0.74 | 3243 | – |
| Array-RQMC, on-policy (MC) | 0.891 | 0.846 | 1.03 | 1.03 | -0.8 | 95% | 1.02 | 3294 | – |
| BPS + Array-RQMC, OIS | 0.576 | 0.594 | 0.466 | 0.458 | -1.0 | 95% | 0.69 | 3194 | – |
| BPS + Array-RQMC, PDIS | 0.576 | 0.594 | 0.466 | 0.458 | -1.0 | 95% | 0.69 | 3194 | – |

## Paired MSE ratios per episode

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| comparison | n = 32 | n = 256 | n = 2048 |
|---|---|---|---|
| BPS + Array-RQMC, OIS vs BPS + OIS (i.i.d.) | 1.01 ~ | 0.941 ~ | 0.868 ~ |
| BPS + Array-RQMC, PDIS vs BPS + PDIS (i.i.d.) | 1.01 ~ | 0.941 ~ | 0.868 ~ |
| BPS + Array-RQMC, OIS vs ROS-MC | 0.616 ✓ | 0.581 ~ | 0.482 ✓ |
| BPS + Array-RQMC, PDIS vs ROS-MC | 0.616 ✓ | 0.581 ~ | 0.482 ✓ |
| Array-RQMC, on-policy (MC) vs OS-MC | 1.01 ~ | 0.763 ~ | 1.06 ~ |
| Array-RQMC, on-policy (MC) vs ROS-MC | 0.952 ~ | 0.827 ~ | 1.07 ~ |
