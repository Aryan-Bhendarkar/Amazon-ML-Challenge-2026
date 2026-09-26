# 20260926-1928_aryan-box2_fr-sample-report

**Hypothesis:** FR norm v2 + French BIZ words: label-free effect on FR test sample; FR density shift

Biz-word up-flips (v2 -> v2biz), sample of 25:

```
    prob   prob_n                                                                                            s1                                                                                               cand
0.371801 0.883365                                  XX Medico | 8 Rue Etienne Marcel, Pessac, Nouvelle-Aquitaine                                   XX Medico Développement | 13 - R Etienne Marcel, Pessac, Gironde
0.746947 0.999711                Lycée Saint Mission | 8 Rue du Professeur Moreau, Bordeaux, Nouvelle-Aquitaine             Lycée Mission Développement | 8 RUE DU PROFESSEUR MOREAU, BORDEAUX, Nouvelle-Aquitaine
0.474982 0.999616                                     Sona Club | 14 Rue Audubert, Bordeaux, Nouvelle-Aquitaine                                            Sona Développement | 14 Rue Audubert, Bordeaux, Gironde
0.061617 0.950406                      Nantes Sportif SAS | 41 Avenue du Clos du Cens, Nantes, Pays de la Loire                               Nantes SAS Et Fils | 41 Av Du Clos Du Cens, Nantes, Loire-Atlantique
0.326942 0.828628                         Deducation & Fils SARL | 8 Rue du Chemin Vert, Lille, Hauts-de-France                                 Deducation & Fils Développement Sàrl | Lille, 9 Rue Du Chemin Vert
0.005399 0.794302                                      MWO Primaire SA | Calais, 80 Rue Vauban, Hauts-de-France                                                           MWO SA Et Fils | 80 - RUE VAUBAN, CALAIS
0.019965 0.855269                   Enfants Culture SARL | 25 Rue de l'Amiral Courbet, Roubaix, Hauts-de-France                                       Enfants SARL + Fils | No 25 Rue De L'amiral Courbet, Roubaix
0.283385 0.998595                          Racines & Fils SARL | 73 rue de la tannerie, Hauts-de-France, Calais                      Racines & SARL-Développement | CALAIS, 73 RUE DE LA TANNERIE, Hauts-de-France
0.526743 0.785634                          Bordeaux Agricole SARL | 17 Rue Tillet, Bordeaux, Nouvelle-Aquitaine                                   Bordeaux àgricole Développement SARL | No. 19 R Tillet, Bordeaux
0.695074 0.809423             Centre Médical du Vincent | 48 Rue Paul Langevin, Saint-Nazaire, Pays de la Loire                            Centre Médical Vincent Développement | 55 Rue Paul Langevin, St-nazaire
0.010205 0.828510                        IY Culturelle SARL | 2 Cours Antonio Vivaldi, Nantes, Pays de la Loire                                  IY SARL-ET FILS | 2 CRS ADTONIO VIVALDI, NANTES, Loire-Atlantique
0.137802 0.844112                                     OCE Ecole SARL | 31 Rue Paul Bert, Lille, Hauts-de-France                            OCE Ecole Développement SARL | LILLE, Hauts-de-France, 38 RUE PAUL BERT
0.055923 0.875999       Lège-Cap-Ferret Culturelle SARL | 27 Route d'Ignac, Lège-Cap-Ferret, Nouvelle-Aquitaine               LÈGE-CAP-FERRET SARL ET FILS | Nouvelle-Aquitaine, LEGE-CAP-FERRET, 27 ROUTE D'IGNAC
0.433415 0.998849             Sports Elektro Union SARL | 30 Rue Jules Guesde, Saint-Herblain, Pays de la Loire             Sports Elektro SARL Développement | Rue Jules Guesde, Saint-herblain, Loire-Atlantique
0.193091 0.957124                     KD Groupement SARL | 111 Route de Marsac, Saint-Nazaire, Pays de la Loire                             KD SARL Et Fils | N°111 Rte De Marsac, Saint-nazaire, Loire-Atlantique
0.013672 0.983146                               Dici Club SAS | 20 Rue Albert Camus, Tourcoing, Hauts-de-France                                       S.A.S DICI PATRIMOINE | Nord, Tourcoing, 20 RUE ALBERT CAMUS
0.766562 0.874079                 Établissements Conscrits SAS | 9 Allée la Chenaie, Pessac, Nouvelle-Aquitaine                  établissements Conscrits Développement Sas | 22 Allee La Chenaie, Pessac, Gironde
0.126871 0.937158                             Dunkerque Club SARL | 39 Rue Voltaire, Dunkerque, Hauts-de-France                                 Dunkerque SARL  & Fils | 39 R VOLTAIRE, Hauts-de-France, DUNKERQUE
0.096821 0.799865         Cdg Anciens SARL | 3 Rue des Freres Montgolfier, La Teste-de-Buch, Nouvelle-Aquitaine Cdg Anciens Développement SARL | 24 R Des Freres Montgolfier, La Teste-de-buch, Nouvelle-Aquitaine
0.487314 0.788591            Democratie Patrimoine EURL | 32 Rue du Moulin des Carmes, Nantes, Pays de la Loire   Democratie  Patrimoine Développement EURL | 35 R. DU MOULIN DES CARMES, NANTES, Pays de la Loire
0.298920 0.915981 XP Amis SARL | Hauts-de-France, 116 Boulevard de la République François Mitterrand, Dunkerque   XP SARL Et Fils | 116 Boulevard De La Republique Francois Mitterrand, Dunkerque, Hauts-de-France
0.743674 0.946199                           Calais Gestion SARL | 75 Route de Coulogne, Calais, Hauts-de-France                                    CALAIS GESTION DÉVELOPPEMENT SARL | 76 Rte. De Coulogne, Calais
0.091224 0.934858                                   Doeuvres Ecole | 10 Chemin Renaud, Nantes, Pays de la Loire                                          DOEUVRES  & FILS | 10 CH RENAUD, NANTES, Loire-Atlantique
0.075294 0.950595              Usagers Federation SARL | 58 Rue du Maréchal Bugeaud, Tourcoing, Hauts-de-France                        Usagers SARL &-Fils | 58 R. DU MARÉCHAL BUGEAUD, TOURCOING, Hauts-de-France
0.055497 0.997708                    Abcd Societe SAS | 86 Rue Charles Longuet, Saint-Nazaire, Pays de la Loire                    Abcd SAS Développement | 86 R. CHARLES LONGUET, SAINT-NAZAIRE, Pays de la Loire
```

Label-free test-sample report (20k S1/country, frozen 0710 repro, t=0.775).

```
               n_s1 pred_per_s1 empty_rate share_1match share_4plus strong_cand_S1 strong_but_empty band_records_per_s1 coloc_S1_nonempty single_addr_S1_nonempty coloc_S1_share     s1_addr_self        s1_addr_c       s1_hs_self      pool_addr_c      s1_name_self             ex_lfrac_cmax             mi_lfrac_cmax            ex_ldf_max ex_biz_rate mi_biz_rate
France/v1     20000       3.215     0.0619       0.0821      0.4156          0.908           0.0368               0.358            0.9397                  0.9377          0.189  [1.0, 1.0, 2.0]  [0.0, 0.0, 1.0]  [1.0, 1.0, 2.0]  [0.0, 1.0, 5.0]  [1.0, 2.0, 20.0]  [-4.971, -3.773, -2.524]  [-4.787, -3.661, -2.524]   [0.0, 3.932, 3.932]      0.1771      0.1133
India/v1      20000       3.281     0.0592       0.0777      0.4345         0.9677           0.0454               0.181            0.9502                  0.9402         0.0642  [1.0, 1.0, 1.0]  [0.0, 0.0, 0.0]  [1.0, 1.0, 1.0]  [0.0, 1.0, 2.0]  [1.0, 2.0, 20.0]  [-7.132, -4.487, -3.595]  [-8.248, -4.548, -3.721]   [0.0, 3.932, 3.932]      0.4323      0.3042
US/v1         20000       3.409     0.0546       0.0666      0.4654         0.9608           0.0434               0.234            0.9528                  0.9449          0.053  [1.0, 1.0, 1.0]  [0.0, 0.0, 1.0]  [1.0, 1.0, 1.0]  [0.0, 0.0, 3.0]  [1.0, 1.0, 20.0]  [-7.694, -5.267, -3.726]   [-7.899, -5.42, -3.726]   [0.0, 3.932, 3.932]      0.2626      0.1329
France/v2     20000       3.229     0.0608       0.0813      0.4222          0.976           0.0465               0.297            0.9418                  0.9385         0.1968  [1.0, 1.0, 2.0]  [0.0, 0.0, 1.0]  [1.0, 1.0, 2.0]  [0.0, 1.0, 6.0]  [1.0, 2.0, 20.0]  [-5.352, -3.874, -2.807]  [-4.971, -3.773, -2.807]  [-1.0, 3.932, 3.932]      0.1227      0.0507
France/v2biz  20000       3.298     0.0586        0.076      0.4404          0.976           0.0447               0.333            0.9433                  0.9409         0.1968  [1.0, 1.0, 2.0]  [0.0, 0.0, 1.0]  [1.0, 1.0, 2.0]  [0.0, 1.0, 6.0]  [1.0, 2.0, 20.0]  [-5.352, -3.874, -2.807]  [-4.971, -3.773, -2.807]  [-1.0, 3.932, 3.932]      0.2148       0.102
```

FR shifts:
```
{
 "v2": {
  "pairs": 1865359,
  "biz_flag_changed": 0.1205,
  "band_pairs": 7617,
  "band_mean_dprob": -0.1113,
  "cross_t_up": 1863,
  "cross_t_down": 1667
 },
 "v2biz": {
  "pairs": 1865359,
  "biz_flag_changed": 0.2402,
  "band_pairs": 7617,
  "band_mean_dprob": -0.082,
  "cross_t_up": 3216,
  "cross_t_down": 1638
 },
 "v2biz_vs_v2": {
  "biz_flag_changed": 0.1389,
  "band_mean_dprob": 0.0381,
  "cross_t_up": 1384,
  "cross_t_down": 2
 }
}
```

