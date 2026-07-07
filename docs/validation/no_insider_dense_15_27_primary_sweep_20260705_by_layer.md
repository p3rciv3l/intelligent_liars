# No-Insider Dense 15-27 Primary Probe Sweep (By-Layer Execution)

Created: `2026-07-05T23:23:29.649884+00:00`

## Inputs

- Dense cache: `artifacts/probe_features/no_insider_dense_15_27_pooled.h5`
- Layers: `15,16,17,18,19,20,21,22,23,24,25,26,27`
- C values: `0.01,0.03,0.1,0.3,1,3,10`
- Task sets: `all_no_insider`, `no_repe`

## Candidate Ranking

| Rank | Result | Task set | C | General bal acc | General AUC | Cross bal acc | Cross AUC | Within bal acc | Within AUC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `no_repe_layer21_c001` | no_repe | 0.0100 | 0.9302 | 0.9752 | 0.6903 | 0.8494 | 0.9469 | 0.9814 |
| 2 | `no_repe_layer21_c003` | no_repe | 0.0300 | 0.9290 | 0.9739 | 0.6906 | 0.8408 | 0.9473 | 0.9811 |
| 3 | `no_repe_layer20_c003` | no_repe | 0.0300 | 0.9259 | 0.9714 | 0.6927 | 0.8385 | 0.9442 | 0.9796 |
| 4 | `no_repe_layer20_c001` | no_repe | 0.0100 | 0.9254 | 0.9722 | 0.6923 | 0.8474 | 0.9407 | 0.9795 |
| 5 | `no_repe_layer19_c001` | no_repe | 0.0100 | 0.9219 | 0.9720 | 0.6961 | 0.8483 | 0.9333 | 0.9779 |
| 6 | `no_repe_layer21_c01` | no_repe | 0.1000 | 0.9211 | 0.9705 | 0.6856 | 0.8330 | 0.9443 | 0.9801 |
| 7 | `no_repe_layer19_c003` | no_repe | 0.0300 | 0.9205 | 0.9715 | 0.6932 | 0.8389 | 0.9375 | 0.9784 |
| 8 | `no_repe_layer22_c001` | no_repe | 0.0100 | 0.9199 | 0.9698 | 0.6813 | 0.8275 | 0.9431 | 0.9794 |
| 9 | `no_repe_layer23_c001` | no_repe | 0.0100 | 0.9195 | 0.9694 | 0.6628 | 0.8135 | 0.9359 | 0.9770 |
| 10 | `no_repe_layer20_c01` | no_repe | 0.1000 | 0.9179 | 0.9678 | 0.6877 | 0.8293 | 0.9433 | 0.9785 |

## All Sweep Jobs

| Result | Task set | C | Tasks | Within | Cross | General | Directions | General directions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `no_repe_layer21_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer21_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer20_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer20_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer19_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer21_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer19_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer22_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer23_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer20_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer22_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer19_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer24_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer23_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer21_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer21_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer21_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer23_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer19_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer25_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer22_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer20_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer18_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer20_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer24_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer22_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer26_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer18_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer21_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer20_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer22_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer18_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer20_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer21_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer23_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer19_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer22_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer22_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer23_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer25_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer19_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer23_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer21_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer24_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer20_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer24_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer19_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer19_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer26_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer20_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer19_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer16_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer18_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer20_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer22_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer21_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer17_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer22_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer24_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer27_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer17_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer21_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer20_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer16_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer19_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer25_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer17_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer22_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer19_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer23_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer26_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer25_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer23_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer26_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer18_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer22_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer24_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer18_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer23_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer25_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer17_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer20_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer27_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer16_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer18_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer18_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer26_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer23_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer24_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer18_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer21_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer15_c001` | no_repe | 0.0100 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer25_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer19_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer15_c003` | no_repe | 0.0300 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer26_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer18_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer21_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer22_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer23_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer20_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer16_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer25_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer17_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer27_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer15_c01` | no_repe | 0.1000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer24_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer22_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer19_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer21_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer17_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer24_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer26_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer18_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer20_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer17_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer25_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer25_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer27_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer19_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer23_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer27_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer24_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer22_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer16_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer26_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer26_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer17_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer24_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer17_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer15_c03` | no_repe | 0.3000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer26_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer25_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer17_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer18_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer17_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer27_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `no_repe_layer16_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer23_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer25_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer16_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer15_c1` | no_repe | 1.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer24_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer26_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer23_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer16_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer15_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer27_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer18_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer27_c3` | no_repe | 3.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer24_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer27_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer16_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer25_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer24_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer16_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer18_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `no_repe_layer15_c10` | no_repe | 10.0000 | 16 | 16 | 240 | 16 | 16 | 1 |
| `all_no_insider_layer26_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer25_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer17_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer16_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer27_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer25_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer26_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer15_c003` | all_no_insider | 0.0300 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer26_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer15_c001` | all_no_insider | 0.0100 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer15_c01` | all_no_insider | 0.1000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer17_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer27_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer16_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer17_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer15_c03` | all_no_insider | 0.3000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer16_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer15_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer27_c1` | all_no_insider | 1.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer16_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer27_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer27_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer15_c3` | all_no_insider | 3.0000 | 17 | 17 | 272 | 17 | 17 | 1 |
| `all_no_insider_layer15_c10` | all_no_insider | 10.0000 | 17 | 17 | 272 | 17 | 17 | 1 |

## Top Layer Configurations

| Rank | Task set | C | Layer | General bal acc | General AUC | Cross bal acc | Cross AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | no_repe | 0.0100 | 21 | 0.9302 | 0.9752 | 0.6903 | 0.8494 |
| 2 | no_repe | 0.0300 | 21 | 0.9290 | 0.9739 | 0.6906 | 0.8408 |
| 3 | no_repe | 0.0300 | 20 | 0.9259 | 0.9714 | 0.6927 | 0.8385 |
| 4 | no_repe | 0.0100 | 20 | 0.9254 | 0.9722 | 0.6923 | 0.8474 |
| 5 | no_repe | 0.0100 | 19 | 0.9219 | 0.9720 | 0.6961 | 0.8483 |
| 6 | no_repe | 0.1000 | 21 | 0.9211 | 0.9705 | 0.6856 | 0.8330 |
| 7 | no_repe | 0.0300 | 19 | 0.9205 | 0.9715 | 0.6932 | 0.8389 |
| 8 | no_repe | 0.0100 | 22 | 0.9199 | 0.9698 | 0.6813 | 0.8275 |
| 9 | no_repe | 0.0100 | 23 | 0.9195 | 0.9694 | 0.6628 | 0.8135 |
| 10 | no_repe | 0.1000 | 20 | 0.9179 | 0.9678 | 0.6877 | 0.8293 |
| 11 | no_repe | 0.0300 | 22 | 0.9161 | 0.9684 | 0.6744 | 0.8182 |
| 12 | no_repe | 0.1000 | 19 | 0.9161 | 0.9683 | 0.6875 | 0.8284 |
| 13 | no_repe | 0.0100 | 24 | 0.9149 | 0.9651 | 0.6572 | 0.7958 |
| 14 | no_repe | 0.0300 | 23 | 0.9148 | 0.9669 | 0.6567 | 0.8056 |
| 15 | no_repe | 0.3000 | 21 | 0.9133 | 0.9639 | 0.6816 | 0.8271 |
| 16 | all_no_insider | 0.0300 | 21 | 0.9120 | 0.9577 | 0.6410 | 0.7576 |
| 17 | all_no_insider | 0.0100 | 21 | 0.9119 | 0.9563 | 0.6410 | 0.7645 |
| 18 | no_repe | 0.1000 | 23 | 0.9115 | 0.9617 | 0.6505 | 0.7986 |
| 19 | no_repe | 0.3000 | 19 | 0.9107 | 0.9629 | 0.6822 | 0.8202 |
| 20 | no_repe | 0.0100 | 25 | 0.9104 | 0.9644 | 0.6443 | 0.7862 |
| 21 | no_repe | 0.1000 | 22 | 0.9099 | 0.9650 | 0.6654 | 0.8099 |
| 22 | no_repe | 0.3000 | 20 | 0.9086 | 0.9620 | 0.6842 | 0.8227 |
| 23 | no_repe | 0.0100 | 18 | 0.9080 | 0.9637 | 0.6737 | 0.8242 |
| 24 | all_no_insider | 0.0300 | 20 | 0.9079 | 0.9569 | 0.6461 | 0.7569 |
| 25 | no_repe | 0.0300 | 24 | 0.9079 | 0.9590 | 0.6474 | 0.7864 |
| 26 | no_repe | 0.3000 | 22 | 0.9075 | 0.9595 | 0.6588 | 0.8040 |
| 27 | no_repe | 0.0100 | 26 | 0.9069 | 0.9573 | 0.6354 | 0.7744 |
| 28 | no_repe | 0.0300 | 18 | 0.9059 | 0.9632 | 0.6737 | 0.8170 |
| 29 | all_no_insider | 0.1000 | 21 | 0.9059 | 0.9541 | 0.6364 | 0.7514 |
| 30 | all_no_insider | 0.0100 | 20 | 0.9057 | 0.9538 | 0.6460 | 0.7642 |

## General-Domain Task Readout

| Task set | Task | Mean bal acc | Mean AUC | Best bal acc | Best result | Best layer | Best C |
|---|---|---:|---:|---:|---|---:|---:|
| no_repe | `geometry_of_truth__best` | 0.9980 | 0.9999 | 0.9994 | `no_repe_layer16_c03` | 16 | 0.3000 |
| all_no_insider | `geometry_of_truth__best` | 0.9978 | 0.9999 | 0.9994 | `all_no_insider_layer16_c03` | 16 | 0.3000 |
| no_repe | `claims__definitional_gemini_600_full` | 0.9459 | 0.9818 | 0.9865 | `no_repe_layer19_c001` | 19 | 0.0100 |
| no_repe | `internal_state__elements` | 0.9361 | 0.9800 | 0.9786 | `no_repe_layer19_c001` | 19 | 0.0100 |
| all_no_insider | `internal_state__elements` | 0.9331 | 0.9763 | 0.9786 | `all_no_insider_layer19_c001` | 19 | 0.0100 |
| no_repe | `claims__evidential_gemini_600_full` | 0.9298 | 0.9777 | 0.9862 | `no_repe_layer21_c001` | 21 | 0.0100 |
| all_no_insider | `claims__definitional_gemini_600_full` | 0.9267 | 0.9714 | 0.9865 | `all_no_insider_layer19_c001` | 19 | 0.0100 |
| no_repe | `internal_state__facts` | 0.9165 | 0.9674 | 0.9870 | `no_repe_layer20_c001` | 20 | 0.0100 |
| no_repe | `internal_state__cities` | 0.9065 | 0.9622 | 0.9480 | `no_repe_layer17_c001` | 17 | 0.0100 |
| no_repe | `internal_state__companies` | 0.9013 | 0.9565 | 0.9333 | `no_repe_layer21_c001` | 21 | 0.0100 |
| all_no_insider | `claims__evidential_gemini_600_full` | 0.9003 | 0.9600 | 0.9690 | `all_no_insider_layer20_c001` | 20 | 0.0100 |
| all_no_insider | `sandbagging_v2__wmdp_mmlu` | 0.8995 | 0.9621 | 1.0000 | `all_no_insider_layer16_c001` | 16 | 0.0100 |
| all_no_insider | `internal_state__cities` | 0.8985 | 0.9541 | 0.9480 | `all_no_insider_layer17_c003` | 17 | 0.0300 |
| no_repe | `sycophancy__mmlu_stem_same_conf_all` | 0.8963 | 0.9349 | 0.9536 | `no_repe_layer26_c1` | 26 | 1.0000 |
| all_no_insider | `internal_state__companies` | 0.8960 | 0.9530 | 0.9333 | `all_no_insider_layer19_c01` | 19 | 0.1000 |
| no_repe | `sandbagging_v2__wmdp_mmlu` | 0.8919 | 0.9639 | 1.0000 | `no_repe_layer16_c001` | 16 | 0.0100 |
| all_no_insider | `sycophancy__mmlu_stem_same_conf_all` | 0.8903 | 0.9324 | 0.9520 | `all_no_insider_layer25_c1` | 25 | 1.0000 |
| no_repe | `claims__logical_gemini_600_full` | 0.8649 | 0.9270 | 0.9306 | `no_repe_layer21_c001` | 21 | 0.0100 |
| no_repe | `claims__fictional_gemini_600_full` | 0.8548 | 0.9314 | 0.9067 | `no_repe_layer21_c003` | 21 | 0.0300 |
| no_repe | `ethics__commonsense` | 0.8541 | 0.9249 | 0.9010 | `no_repe_layer21_c001` | 21 | 0.0100 |
| all_no_insider | `ethics__commonsense` | 0.8499 | 0.9218 | 0.9000 | `all_no_insider_layer21_c003` | 21 | 0.0300 |
| all_no_insider | `internal_state__facts` | 0.8495 | 0.9187 | 0.9545 | `all_no_insider_layer20_c001` | 20 | 0.0100 |
| no_repe | `geometry_of_truth__mixed` | 0.8470 | 0.9229 | 0.8692 | `no_repe_layer20_c001` | 20 | 0.0100 |
| all_no_insider | `internal_state__inventions` | 0.8470 | 0.9136 | 0.9304 | `all_no_insider_layer16_c001` | 16 | 0.0100 |
| all_no_insider | `claims__logical_gemini_600_full` | 0.8454 | 0.9124 | 0.9410 | `all_no_insider_layer21_c001` | 21 | 0.0100 |
| all_no_insider | `geometry_of_truth__mixed` | 0.8452 | 0.9198 | 0.8683 | `all_no_insider_layer19_c001` | 19 | 0.0100 |
| no_repe | `internal_state__animals` | 0.8444 | 0.9319 | 0.8929 | `no_repe_layer20_c001` | 20 | 0.0100 |
| no_repe | `internal_state__inventions` | 0.8370 | 0.9111 | 0.9261 | `no_repe_layer16_c001` | 16 | 0.0100 |
| all_no_insider | `internal_state__animals` | 0.8362 | 0.9225 | 0.8968 | `all_no_insider_layer22_c001` | 22 | 0.0100 |
| all_no_insider | `claims__fictional_gemini_600_full` | 0.8346 | 0.9118 | 0.9233 | `all_no_insider_layer21_c001` | 21 | 0.0100 |
| no_repe | `roleplaying__plain` | 0.7863 | 0.8629 | 0.9272 | `no_repe_layer21_c003` | 21 | 0.0300 |
| all_no_insider | `roleplaying__plain` | 0.7860 | 0.8636 | 0.9279 | `all_no_insider_layer21_c01` | 21 | 0.1000 |
| all_no_insider | `repe_honesty__IF_dishonest` | 0.5977 | 0.6191 | 0.7051 | `all_no_insider_layer20_c01` | 20 | 0.1000 |

## Selection Rule

Prefer configs with high general-domain balanced accuracy and AUC, then cross-task metrics, then stability across adjacent layers. Do not select a direction from within-task accuracy alone.
