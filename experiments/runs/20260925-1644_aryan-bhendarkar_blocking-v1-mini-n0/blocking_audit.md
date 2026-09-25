# Blocking audit (mini, cands=data/cands/v1/mini_n0_pairs.parquet)

true pairs 304,592 · missed 7,485 (0.0246) · recall 0.9754

## Name bucket (exclusive) × country
```
shape: (6, 5)
┌───────────────────────┬──────┬───────┬───────┬────────────┐
│ bucket                ┆ US   ┆ India ┆ total ┆ recall_pts │
│ ---                   ┆ ---  ┆ ---   ┆ ---   ┆ ---        │
│ str                   ┆ u32  ┆ u32   ┆ u32   ┆ f64        │
╞═══════════════════════╪══════╪═══════╪═══════╪════════════╡
│ native_script         ┆ 0    ┆ 3261  ┆ 3261  ┆ 0.0107     │
│ name_near_identical   ┆ 1048 ┆ 1025  ┆ 2073  ┆ 0.0068     │
│ name_typo_moderate    ┆ 507  ┆ 439   ┆ 946   ┆ 0.0031     │
│ name_heavy_diff       ┆ 268  ┆ 334   ┆ 602   ┆ 0.002      │
│ name_extra_or_reorder ┆ 198  ┆ 182   ┆ 380   ┆ 0.0012     │
│ domain_handle         ┆ 70   ┆ 153   ┆ 223   ┆ 0.0007     │
└───────────────────────┴──────┴───────┴───────┴────────────┘
```

## Name bucket × address state of the candidate
```
shape: (6, 6)
┌────────────────────┬───────────────┬────────────────┬────────────┬─────────────┬─────────────────┐
│ bucket             ┆ cand_no_house ┆ house_zero_pad ┆ house_diff ┆ house_equal ┆ cand_addr_empty │
│ ---                ┆ ---           ┆ ---            ┆ ---        ┆ ---         ┆ ---             │
│ str                ┆ u32           ┆ u32            ┆ u32        ┆ u32         ┆ u32             │
╞════════════════════╪═══════════════╪════════════════╪════════════╪═════════════╪═════════════════╡
│ name_near_identica ┆ 74            ┆ 72             ┆ 341        ┆ 208         ┆ 1378            │
│ l                  ┆               ┆                ┆            ┆             ┆                 │
│ domain_handle      ┆ 20            ┆ 25             ┆ 97         ┆ 80          ┆ 1               │
│ native_script      ┆ 266           ┆ 204            ┆ 1245       ┆ 1516        ┆ 30              │
│ name_extra_or_reor ┆ 27            ┆ 16             ┆ 81         ┆ 58          ┆ 198             │
│ der                ┆               ┆                ┆            ┆             ┆                 │
│ name_heavy_diff    ┆ 76            ┆ 45             ┆ 235        ┆ 158         ┆ 88              │
│ name_typo_moderate ┆ 38            ┆ 43             ┆ 201        ┆ 153         ┆ 511             │
└────────────────────┴───────────────┴────────────────┴────────────┴─────────────┴─────────────────┘
```

## Stratified sample
```
shape: (40, 9)
┌─────────┬─────┬───────────────────────┬─────────────────┬───┬─────────────────────────────┬──────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┬─────────────────────────────────┐
│ country ┆ src ┆ bucket                ┆ addr_state      ┆ … ┆ n_core                      ┆ n_core_c                         ┆ a_full                                                                  ┆ a_full_c                        │
│ ---     ┆ --- ┆ ---                   ┆ ---             ┆   ┆ ---                         ┆ ---                              ┆ ---                                                                     ┆ ---                             │
│ str     ┆ str ┆ str                   ┆ str             ┆   ┆ str                         ┆ str                              ┆ str                                                                     ┆ str                             │
╞═════════╪═════╪═══════════════════════╪═════════════════╪═══╪═════════════════════════════╪══════════════════════════════════╪═════════════════════════════════════════════════════════════════════════╪═════════════════════════════════╡
│ India   ┆ S2  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ jay consulting              ┆ jay consulting                   ┆ c 380 first fl palam sec 7 dwarka s w delhi dl                          ┆                                 │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ golden infra                ┆ goldn inphra                     ┆ 85 nr bsf 84 85 sai spring feild layout vaderapura ynk bangalore ka     ┆ 85 bangalore krnatk             │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ ss investments              ┆ eses investmemts                 ┆ 72 1st main 1st ph w of chord rd manjunatha ngr nr swathi restaurant r… ┆ ka mumbai bangalore 72          │
│ India   ┆ S2  ┆ native_script         ┆ house_diff      ┆ … ┆ jai maa products            ┆ jai ma prodkts                   ┆ pnr col jana chaitanya ph 2 ameenpur medak 70 hyderabad tg              ┆ 70 hyderabad tg                 │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ modern logistics            ┆ modrn lojistiks                  ┆ 14e 12a fl 5 78 commerce centre pandit madam mohan malviya marg tardeo… ┆ 14e mumbai suburban mharastr    │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ sai business                ┆ sai bijnes                       ┆ ismail bldg dr ambedkar rd mh 23 mumbai                                 ┆ 23 mumbai mharastr              │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ life management             ┆ laiph mainejmemt                 ┆ shop 73 30 g f vardhman market plaza c c pitampura delhi delhi w delhi… ┆ shop 73 n w dl                  │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ universal foundation        ┆ yunivrcl kpvuntesn piraivet      ┆ g1 amirtha varshini 131 old 52 kothanda ramar koil st w mambala m chen… ┆ gd 1 chennai tn                 │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ best projects               ┆ pest purajekts piraivet          ┆ 7 306 a durga ngr vanniampatti villakku srivilliputhur chennai tn       ┆ 7 306 a chennai tmilnatu        │
│ India   ┆ S2  ┆ native_script         ┆ house_diff      ┆ … ┆ perfect healthcare          ┆ prphekt heltker                  ┆ ka bangalore 1st main gm palya new thippasandra opp vaibhav residency … ┆ 02 n a bangalore krnatk         │
│ India   ┆ S3  ┆ domain_handle         ┆ house_equal     ┆ … ┆ al technologies             ┆ altechnologies                   ┆ b 12 parmar trade centre sr 10 10a 12 cannaug pune mh                   ┆ mh pune b 12                    │
│ India   ┆ S3  ┆ name_extra_or_reorder ┆ house_equal     ┆ … ┆ rj services                 ┆ rj srliies                       ┆ 144 1st fl powai plaza hiranandani gardens powai mumbai mumbai city mh  ┆ 144 mumbai city mumbai mh       │
│ India   ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ aditya industries           ┆ dr aditya industries enterprises ┆ office 5 and 6 ground fl dda market naraina rd pandav ngr opp janki da… ┆                                 │
│ India   ┆ S3  ┆ name_typo_moderate    ┆ house_zero_pad  ┆ … ┆ kuldeep allied              ┆ kuldeep ahlemid                  ┆ 59e tiljala rd karaya kolkata kolkata howrah wb                         ┆ 059e kolkata wb                 │
│ India   ┆ S3  ┆ name_typo_moderate    ┆ house_diff      ┆ … ┆ united management           ┆ united mscagemet                 ┆ 8 12 38 vrindavan ngr st 8 habsiguda hyderabad tg                       ┆ b3 8 12 38 hyderabad telmgan    │
│ India   ┆ S3  ┆ native_script         ┆ cand_no_house   ┆ … ┆ hitech consultancy          ┆ haitek consultancy               ┆ c o rishu singh sn 115 1 herambh co op housing pashan pune mh           ┆ c o rishu singh pune mh         │
│ India   ┆ S3  ┆ native_script         ┆ house_equal     ┆ … ┆ krishna consulting          ┆ krsn knsltimg                    ┆ 404 407 ashvmegh aligance bhudarpura rd hirabag ambawadi ahmedabad gj   ┆ ahmedabad 404 gj                │
│ India   ┆ S3  ┆ native_script         ┆ house_diff      ┆ … ┆ premier producer            ┆ primiyr prodyusr limirrd         ┆ kuzhivelippady ste 16s a sq 1st fl edathala p o ernakulam kl 14 291 m   ┆ 14 291 m ernakulam keralam      │
│ India   ┆ S3  ┆ native_script         ┆ house_diff      ┆ … ┆ classic software            ┆ klasik sophtveyr                 ┆ 210 mumbai city fl 2 416 hammersmith ind estate narayan pathare marg o… ┆ 10 mahim mumbai city mh         │
│ India   ┆ S3  ┆ native_script         ┆ house_diff      ┆ … ┆ first energy                ┆ phrst enrji                      ┆ a 76 samarth ngr dd ngr gwalior mp                                      ┆ a 46 gwalior mp                 │
│ US      ┆ S2  ┆ name_extra_or_reorder ┆ house_diff      ┆ … ┆ national bnb                ┆ national center                  ┆ 940 russell st unit c nashville tn                                      ┆ 94 russell st nashville tn      │
│ US      ┆ S2  ┆ name_extra_or_reorder ┆ cand_no_house   ┆ … ┆ southern bio                ┆ southern services                ┆ 1102 sumner ave indianapolis in                                         ┆ sumner ave inianapolis in       │
│ US      ┆ S2  ┆ name_heavy_diff       ┆ cand_no_house   ┆ … ┆ onyx                        ┆ 0ycnxf                           ┆ mesa unit 519 2550 ellsworth rd az                                      ┆ ellsworth rd mesa az            │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_zero_pad  ┆ … ┆ pediatric dental associates ┆ pediatric dental associates      ┆ 5063 stonehill ln stallings nc                                          ┆ 005063 stonehill ln matthews nc │
│ US      ┆ S2  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ optva                       ┆ optva enterprises                ┆ 275 glen ct town of new haven wi                                        ┆                                 │
│ US      ┆ S2  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ obrien malt islip           ┆ obrien malt islip                ┆ 2 suydam ln islip ny                                                    ┆                                 │
│ US      ┆ S2  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ urgent care physicians      ┆ urgent care physicians           ┆ 2027 mcbride rd yellville ar                                            ┆                                 │
│ US      ┆ S2  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ clear medical               ┆ clear mredsical                  ┆ 3981 mikehill dr hamilton oh                                            ┆                                 │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ health institute            ┆ health instetue                  ┆ 10823 91st ave e bixby ok                                               ┆ 10823d 91st ave e tulsa ok      │
│ US      ┆ S2  ┆ name_typo_moderate    ┆ cand_addr_empty ┆ … ┆ congregation emanuel        ┆ congregation eaonrule            ┆ 513 spring garden st lower gwynedd township pa                          ┆                                 │
│ US      ┆ S3  ┆ domain_handle         ┆ house_diff      ┆ … ┆ adams innovative trg        ┆ trginnovativeadams               ┆ 1015 26th st parsons ks                                                 ┆ ks parrsons 26rd st             │
│ US      ┆ S3  ┆ name_extra_or_reorder ┆ house_diff      ┆ … ┆ valley charities            ┆ valley services                  ┆ 2914 61st dr phoenix az                                                 ┆ 4914 61st dr phoeinx az         │
│ US      ┆ S3  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ blue bike shop              ┆ blue bike sh0p                   ┆ 38 sterling st fishkill ny                                              ┆ 38d sterling st beacon ny       │
│ US      ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ beacon consumer enterprises ┆ beacon cnosumler enterprises     ┆ 36 otter ave unit apt 33 salem city va                                  ┆                                 │
│ US      ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ cardiology health           ┆ cardiology health                ┆ nc 482 ab elliott rd pg                                                 ┆                                 │
│ US      ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ family partners             ┆ fami1y partners                  ┆ 415 tucson ave flagstaff az                                             ┆                                 │
│ US      ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ empire project              ┆ empire project trading           ┆ 253 billings st quincy ma                                               ┆                                 │
│ US      ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ cardiology health           ┆ cardi0logy health                ┆ 620 westmore dr indianapolis in                                         ┆                                 │
│ US      ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ certified fresenius         ┆ certified fresenls               ┆ 3201 lanewood dr richmond city va                                       ┆                                 │
│ US      ┆ S3  ┆ name_typo_moderate    ┆ cand_addr_empty ┆ … ┆ national pediatrics         ┆ national pdettrisc               ┆ 1152 canvasback ln denton md                                            ┆                                 │
└─────────┴─────┴───────────────────────┴─────────────────┴───┴─────────────────────────────┴──────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┴─────────────────────────────────┘
```
