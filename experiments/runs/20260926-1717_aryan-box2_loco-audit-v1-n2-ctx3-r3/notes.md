# 20260926-1717_aryan-box2_loco-audit-v1-n2-ctx3-r3

**Hypothesis:** LOCO feature-group audit of the 0710 feature set

LOCO audit, seeds [42, 43, 44] (+ref []). full avg_tuned per seed [0.94207, 0.94243, 0.94277] (seed spread 0.00070).

```
                        variant  seed  n_feats  avg_tuned  d_avg_tuned  avg_src_t  d_avg_src_t  India->US_d  US->India_d
                      -G3_lfrac    42       79    0.94564      0.00357    0.94331      0.00268      0.01146     -0.00431
combo:-B_tok+-G1_pool+-G3_lfrac    43       72    0.94488      0.00245    0.94146      0.00202      0.00981     -0.00491
                      -G3_lfrac    44       79    0.94521      0.00244    0.94015      0.00041      0.00899     -0.00411
combo:-B_tok+-G1_pool+-G3_lfrac    44       72    0.94495      0.00218    0.94083      0.00109      0.00964     -0.00528
combo:-B_tok+-G1_pool+-G3_lfrac    42       72    0.94396      0.00189    0.94034     -0.00029      0.01079     -0.00699
                      -G3_lfrac    43       79    0.94366      0.00123    0.93941     -0.00003      0.00863     -0.00617
          combo:-B_tok+-G1_pool    42       80    0.94279      0.00072    0.94110      0.00047     -0.00039      0.00185
          combo:-B_tok+-G1_pool    43       80    0.94275      0.00032    0.94151      0.00207     -0.00126      0.00190
                           full    44       87    0.94277      0.00000    0.93974      0.00000          NaN          NaN
                           full    43       87    0.94243      0.00000    0.93944      0.00000          NaN          NaN
                           full    42       87    0.94207      0.00000    0.94063      0.00000          NaN          NaN
          combo:-B_tok+-G1_pool    44       80    0.94170     -0.00107    0.94047      0.00073     -0.00198     -0.00015
```

