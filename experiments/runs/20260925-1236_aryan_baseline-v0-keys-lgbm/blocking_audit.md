# Blocking audit (mini, cands=artifacts/20260925-1236_aryan_baseline-v0-keys-lgbm/val_pred.parquet)

true pairs 304,592 · missed 42,456 (0.1394) · recall 0.8606

## Name bucket (exclusive) × country
```
shape: (8, 5)
┌───────────────────────┬───────┬───────┬───────┬────────────┐
│ bucket                ┆ US    ┆ India ┆ total ┆ recall_pts │
│ ---                   ┆ ---   ┆ ---   ┆ ---   ┆ ---        │
│ str                   ┆ u32   ┆ u32   ┆ u32   ┆ f64        │
╞═══════════════════════╪═══════╪═══════╪═══════╪════════════╡
│ name_near_identical   ┆ 13741 ┆ 5599  ┆ 19340 ┆ 0.0635     │
│ native_script         ┆ 0     ┆ 9687  ┆ 9687  ┆ 0.0318     │
│ name_typo_moderate    ┆ 2964  ┆ 2550  ┆ 5514  ┆ 0.0181     │
│ name_heavy_diff       ┆ 1300  ┆ 1493  ┆ 2793  ┆ 0.0092     │
│ name_extra_or_reorder ┆ 1557  ┆ 1044  ┆ 2601  ┆ 0.0085     │
│ domain_handle         ┆ 1456  ┆ 904   ┆ 2360  ┆ 0.0077     │
│ alias                 ┆ 45    ┆ 115   ┆ 160   ┆ 0.0005     │
│ empty_name            ┆ 1     ┆ 0     ┆ 1     ┆ 0.0        │
└───────────────────────┴───────┴───────┴───────┴────────────┘
```

## Name bucket × address state of the candidate
```
shape: (8, 6)
┌────────────────────┬────────────┬─────────────┬────────────────┬─────────────────┬───────────────┐
│ bucket             ┆ house_diff ┆ house_equal ┆ house_zero_pad ┆ cand_addr_empty ┆ cand_no_house │
│ ---                ┆ ---        ┆ ---         ┆ ---            ┆ ---             ┆ ---           │
│ str                ┆ u32        ┆ u32         ┆ u32            ┆ u32             ┆ u32           │
╞════════════════════╪════════════╪═════════════╪════════════════╪═════════════════╪═══════════════╡
│ domain_handle      ┆ 1322       ┆ 190         ┆ 291            ┆ 8               ┆ 549           │
│ alias              ┆ 81         ┆ 13          ┆ 0              ┆ 0               ┆ 66            │
│ name_near_identica ┆ 9421       ┆ 1330        ┆ 1947           ┆ 2533            ┆ 4109          │
│ l                  ┆            ┆             ┆                ┆                 ┆               │
│ native_script      ┆ 5605       ┆ 1346        ┆ 582            ┆ 30              ┆ 2124          │
│ name_extra_or_reor ┆ 1285       ┆ 183         ┆ 244            ┆ 316             ┆ 573           │
│ der                ┆            ┆             ┆                ┆                 ┆               │
│ name_heavy_diff    ┆ 1439       ┆ 251         ┆ 284            ┆ 147             ┆ 672           │
│ empty_name         ┆ 1          ┆ 0           ┆ 0              ┆ 0               ┆ 0             │
│ name_typo_moderate ┆ 2760       ┆ 410         ┆ 523            ┆ 683             ┆ 1138          │
└────────────────────┴────────────┴─────────────┴────────────────┴─────────────────┴───────────────┘
```

## Stratified sample
```
shape: (60, 9)
┌─────────┬─────┬───────────────────────┬─────────────────┬───┬───────────────────────────────────┬──────────────────────────────────┬──────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┐
│ country ┆ src ┆ bucket                ┆ addr_state      ┆ … ┆ n_core                            ┆ n_core_c                         ┆ a_full                                                       ┆ a_full_c                                                     │
│ ---     ┆ --- ┆ ---                   ┆ ---             ┆   ┆ ---                               ┆ ---                              ┆ ---                                                          ┆ ---                                                          │
│ str     ┆ str ┆ str                   ┆ str             ┆   ┆ str                               ┆ str                              ┆ str                                                          ┆ str                                                          │
╞═════════╪═════╪═══════════════════════╪═════════════════╪═══╪═══════════════════════════════════╪══════════════════════════════════╪══════════════════════════════════════════════════════════════╪══════════════════════════════════════════════════════════════╡
│ India   ┆ S2  ┆ domain_handle         ┆ house_diff      ┆ … ┆ shiva management                  ┆ shivamanagement                  ┆ 215 milan garment hub premises milan subway rd mumbai mh     ┆ 15 milan garment hub premises milan subway rd mumbai mh      │
│ India   ┆ S2  ┆ name_near_identical   ┆ cand_no_house   ┆ … ┆ shiv consultants                  ┆ shiv consultants                 ┆ nirankari seeds company chauhan ngr sirsa rd mandi dabwali   ┆ nirankari seeds company chauhan ngr sirsa rd mandi dabwali   │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ sirsa hr                                                     ┆ hr                                                           │
│ India   ┆ S2  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ tulip trading                     ┆ tu1ip trading                    ┆ continental plaza 3g 3rd fl 705 anna salai thousand lights   ┆ continental plaza 4 3g 3rd fl 705 anna salai thousand lights │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ chennai tn                                                   ┆ chennai t…                                                   │
│ India   ┆ S2  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ north exports                     ┆ dr north exports                 ┆ midc am 3 nr sartak hotel madha solapur madha solapur mh     ┆                                                              │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ great foundation                  ┆ gret phaumdesn                   ┆ 37 3rd fl shreshtha vihar dl e delhi dl                      ┆ 37 dl e delhi dl                                             │
│ India   ┆ S2  ┆ native_script         ┆ cand_no_house   ┆ … ┆ krishna supreme construction      ┆ krsn suprim knstraksn praibhet   ┆ c o sankar das vill alangiri alangiri e midnapore wb         ┆ c o sankar das vill alangiri alangiri e midnapore wb         │
│ India   ┆ S2  ┆ native_script         ┆ cand_no_house   ┆ … ┆ indian blue tech                  ┆ intyn blu tek praivrr limirrd    ┆ kottayam c o moosakutty v k kl v publishers bldg kottayam    ┆ c o moosakutty v k v publishers bldg kottayam keralam        │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ real tech                         ┆ riyl tek                         ┆ 93 ground fl village munirka dl s w delhi dl                 ┆ 93 dl s w delhi dl                                           │
│ India   ┆ S2  ┆ native_script         ┆ house_diff      ┆ … ┆ sky constructions                 ┆ skai knstrksns praivrr limirrd   ┆ kl first fl soni steel corporation ernakulam                 ┆ block d 993 first fl soni steel corporation kl ernakulam     │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ bharat international              ┆ bhart imtrnesnl                  ┆ 74 fl 7 274 dnyan sagar rskb rd pinto villa portugese church ┆ mumbai mh 74 mumbai city                                     │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ dadar w m…                                                   ┆                                                              │
│ India   ┆ S2  ┆ native_script         ┆ house_equal     ┆ … ┆ lakshmi producer                  ┆ lksmi prodyusr                   ┆ bldg 17 101 first fl bhanot yusuf sarai community centre dl  ┆ bldg 17 dl dl                                                │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ dl                                                           ┆                                                              │
│ India   ┆ S2  ┆ native_script         ┆ house_diff      ┆ … ┆ jai south media                   ┆ jy sauth midia praibhet          ┆ 31 p khata 54 pokhariput bhubaneswar khordha od              ┆ block d 600 31 p bhubaneswar khordha od                      │
│ India   ┆ S2  ┆ native_script         ┆ house_diff      ┆ … ┆ hitech energy                     ┆ haitek enrji                     ┆ 2 128 10 main rd sf 550 3a2 avanashi coimbatore tn           ┆ 143 2 128 10 main rd sf 550 3a2 avanashi coimbatore tn       │
│ India   ┆ S2  ┆ native_script         ┆ house_diff      ┆ … ┆ unique eastern technology         ┆ yunik istrn teknoloji            ┆ opp reliance petrol pump nr singla dharamkanta sirsa rd      ┆ 228 opp reliance petrol pump nr singla dharamkanta fatehabad │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ fatehabad hr                                                 ┆ hr                                                           │
│ India   ┆ S2  ┆ native_script         ┆ cand_no_house   ┆ … ┆ lakshmi technologies              ┆ lksmi teknolojis                 ┆ opp surpur bus stand at surpur idar sabar kantha gj          ┆ opp surpur bus stand at surpur idar sabar kantha gj          │
│ India   ┆ S3  ┆ domain_handle         ┆ house_diff      ┆ … ┆ jra sciences                      ┆ sciencesprivate                  ┆ prd bldg 1st fl kuriannoor p o kuriannoor pathanamthitta kl  ┆ 397 prd buidling 1st fl kuriannoor p o kuriannoor            │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆                                                              ┆ pathanamthitta kerlm                                         │
│ India   ┆ S3  ┆ domain_handle         ┆ house_equal     ┆ … ┆ pioneer solutions                 ┆ pioneersolutions                 ┆ dl 416 dl dl g f front side portion bhera enclave paschim    ┆ 416 dl dl dl                                                 │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ vihar                                                        ┆                                                              │
│ India   ┆ S3  ┆ name_extra_or_reorder ┆ house_equal     ┆ … ┆ premier properties                ┆ premier services                 ┆ 5th fl raheja centre point 294 c s t rd kalina santacruz e   ┆ 5th fl mumbai mh                                             │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ mumbai mumb…                                                 ┆                                                              │
│ India   ┆ S3  ┆ name_extra_or_reorder ┆ house_diff      ┆ … ┆ perfect products                  ┆ perfect pacst                    ┆ tg g03 alm parkview apartments santoshnagar col hyderabad    ┆ hn 774 g03 alm parkview apartments santoshnagar col          │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆                                                              ┆ hyderabad ranga re…                                          │
│ India   ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ my technologies                   ┆ my technologies enterprises      ┆ dharur rd majalgaon beed mh                                  ┆                                                              │
│ India   ┆ S3  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ sai producer                      ┆ sai producer                     ┆ trivandrum kl kheys medicare mannaniyya college rd pangodu   ┆ block e 419 kheys medicare mannaniyya college rd pangodu     │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ pangode ned…                                                 ┆ pangode nedum…                                               │
│ India   ┆ S3  ┆ name_typo_moderate    ┆ cand_no_house   ┆ … ┆ pz loans                          ┆ pz l0ans                         ┆ icon centre 2nd fl 144 129 nelson manickam rd aminjikarai    ┆ icon centre chennai madras tn                                │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ chennai tn                                                   ┆                                                              │
│ India   ┆ S3  ┆ name_typo_moderate    ┆ house_diff      ┆ … ┆ green care                        ┆ green ccr                        ┆ mh sn 17 4 5 pune bramha angan nr ambavatika kondhawa kd     ┆ 24 sn 17 4 5 bramha angan nr ambavatika kondhawa kd pune     │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆                                                              ┆ mharastr                                                     │
│ India   ┆ S3  ┆ native_script         ┆ cand_no_house   ┆ … ┆ all healthcare                    ┆ al heltker                       ┆ soppinapete village gudibande gp chik ballapur               ┆ soppinapete village gudibande gp kolar chikkaballapur ka     │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ chikkaballapur ka                                            ┆                                                              │
│ India   ┆ S3  ┆ native_script         ┆ house_diff      ┆ … ┆ indian logistics                  ┆ imdiyn lojistiks                 ┆ shop 13 prashwakrupa appt v s marg virar thane mh            ┆ shp 13 5 navi mumbai region thane mh                         │
│ India   ┆ S3  ┆ native_script         ┆ house_diff      ┆ … ┆ fortune investments               ┆ phorcyun investmemts             ┆ 3 17 2nd fl e patel ngr dl central delhi dl                  ┆ 2nd fl e patel ngr dl dl central delhi 3 17 4                │
│ India   ┆ S3  ┆ native_script         ┆ house_equal     ┆ … ┆ one finance                       ┆ vn phainems                      ┆ 1497 dl s delhi dl 1st fl bhardwaj bhawan bhisham pitamah    ┆ 1497 dl s e dl                                               │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ marg kotka m…                                                ┆                                                              │
│ India   ┆ S3  ┆ native_script         ┆ house_diff      ┆ … ┆ city east construction            ┆ citti ist knstrksn piraivet      ┆ block 2 shedii 12 sidcoelecronic complex thiruvika           ┆ 639 block 2 tmilnatu chennai 32 chennai                      │
│         ┆     ┆                       ┆                 ┆   ┆                                   ┆                                  ┆ industrial estate g…                                         ┆                                                              │
│ India   ┆ S3  ┆ native_script         ┆ house_zero_pad  ┆ … ┆ international management          ┆ intrnesnl menejment piraivet     ┆ 5 bharathinagar extn gandhinagar vellore tn                  ┆ 05 chennai city region vellore tn                            │
│ India   ┆ S3  ┆ native_script         ┆ house_diff      ┆ … ┆ international modern construction ┆ imtrnesnl modern construction    ┆ f 304 the address township indore mp                         ┆ indore mp the address township indore 97 f 304               │
│ US      ┆ S2  ┆ name_near_identical   ┆ cand_no_house   ┆ … ┆ wilmington community church       ┆ wilmingt0n community church      ┆ 2016 naamans rd unit unit e3 wilmington de                   ┆ naamans rd wilmington de                                     │
│ US      ┆ S2  ┆ name_near_identical   ┆ cand_no_house   ┆ … ┆ wells strategic diversified       ┆ wells strbtric diversified       ┆ 8107 sandy glen ln houston tx                                ┆ sandy glen ln houston tx                                     │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_zero_pad  ┆ … ┆ route 72 deli                     ┆ route 72                         ┆ 1171 pinecrest cir topeka ks                                 ┆ 001171 pinecrest cir topeka ks                               │
│ US      ┆ S2  ┆ name_near_identical   ┆ cand_no_house   ┆ … ┆ ember                             ┆ ember                            ┆ 2 kopernick rd amsterdam ny                                  ┆ ny kopernick rd amsterdam                                    │
│ US      ┆ S2  ┆ name_near_identical   ┆ cand_no_house   ┆ … ┆ cardiology care                   ┆ cardiology care                  ┆ 14725 mozart ave robbins il                                  ┆ mozart ave robbins il                                        │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_zero_pad  ┆ … ┆ holy catholic church              ┆ holy catholic church             ┆ 103 pinewood dr altona ny                                    ┆ altona ny 00103 pinewood dr                                  │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ regional charities                ┆ regiona1 charities               ┆ 129 paxton ave salt lake city ut                             ┆ 12 paxton ave salt lake city ut                              │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_zero_pad  ┆ … ┆ foot ankle specialists woodbury   ┆ foot ankle specialists           ┆ mn 1020 drew dr woodbury                                     ┆ 01020 drew dr woodbury mn                                    │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ scholarship league leasburg       ┆ scholarship league               ┆ 313 bunny rabbit rd leasburg nc                              ┆ 315 bunny rabbit rd leasburg nc                              │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ national alliance                 ┆ national alliance                ┆ 2345 main st unit 49 mesa az                                 ┆ 234 main st mesa mountain view az                            │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ historical ministries             ┆ historical ministries            ┆ 1362 2000 vernal ut                                          ┆ 136 2000 vernal ut                                           │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ first methodist church            ┆ first methodist                  ┆ 8126 400 laotto in                                           ┆ 812 400 laotto dcp in                                        │
│ US      ┆ S2  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ gulf society                      ┆ gulf society                     ┆ 512 seneca green way fairfax county va                       ┆ 12 seneca green way fairfax county va                        │
│ US      ┆ S2  ┆ name_typo_moderate    ┆ house_diff      ┆ … ┆ western praetorian                ┆ western services                 ┆ 341 forest ave roselle il                                    ┆ 41 forest ave roselle il                                     │
│ US      ┆ S2  ┆ name_typo_moderate    ┆ house_diff      ┆ … ┆ johnson fusion                    ┆ johnson center                   ┆ 1219 spinnaker way sugar land tx                             ┆ 1219 1221 spinnaker way sugar land cdp tx                    │
│ US      ┆ S3  ┆ domain_handle         ┆ house_diff      ┆ … ┆ liberty council                   ┆ libertycouncil                   ┆ va 8630 cartwright ct manassas park city                     ┆ 630 cartwright ct manassas va                                │
│ US      ┆ S3  ┆ name_extra_or_reorder ┆ house_diff      ┆ … ┆ piedmont modern nmp               ┆ piedmont nmp services            ┆ 3222 5th ave chattanooga tn                                  ┆ chattanooga tn 5th ave                                       │
│ US      ┆ S3  ┆ name_extra_or_reorder ┆ house_diff      ┆ … ┆ martinez defense                  ┆ martinez services                ┆ 120 arrowhead st unit 1 fort worth tx                        ┆ ft wroth unit 1 tx 120 arrowhead st                          │
│ US      ┆ S3  ┆ name_extra_or_reorder ┆ house_diff      ┆ … ┆ carlson industries                ┆ carlson services                 ┆ 17900 hayden rd unit unit 1068 scottsdale az                 ┆ scottsdale 7900 hayden rd az                                 │
│ US      ┆ S3  ┆ name_heavy_diff       ┆ cand_no_house   ┆ … ┆ highland choice                   ┆ nexfaye                          ┆ frisco tx 9514 enmore ln                                     ┆ tx enmore ln frisco cdp                                      │
│ US      ┆ S3  ┆ name_heavy_diff       ┆ house_diff      ┆ … ┆ great logistics                   ┆ great center                     ┆ 14194 mineral springs w fork ar                              ┆ 1419 mineral springs w fork ar                               │
│ US      ┆ S3  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ ear nose throat group             ┆ ear nose throat group            ┆ 46 pleasant st southampton ma                                ┆ ma pmb 3454 46 pleasant st southampton                       │
│ US      ┆ S3  ┆ name_near_identical   ┆ cand_addr_empty ┆ … ┆ big hair studio                   ┆ big hair studio                  ┆ 445 grantham st vail az                                      ┆                                                              │
│ US      ┆ S3  ┆ name_near_identical   ┆ cand_no_house   ┆ … ┆ womens health trusted associates  ┆ womens health trusted associates ┆ 189 roosevelt ave tn elizabethton                            ┆ roosevelt ave elizabethton tn                                │
│ US      ┆ S3  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ rev it up hair studio             ┆ studio it up hr rev              ┆ 401 westinghouse rd unit 10307 georgetown tx                 ┆ 401 405 westinghouse rd georgetown tx                        │
│ US      ┆ S3  ┆ name_near_identical   ┆ house_diff      ┆ … ┆ mcmillan cornerstone alchemy      ┆ mcmillan crnerstne alchemy       ┆ 2807 3rd ave atlanta ga                                      ┆ 2808 3rd ave ga atlanta                                      │
│ US      ┆ S3  ┆ name_near_identical   ┆ house_zero_pad  ┆ … ┆ first wealth ventures             ┆ first wealth ventures            ┆ 125 beckview dr montgomery al                                ┆ 00125 beckview dr montgomery township al                     │
│ US      ┆ S3  ┆ name_typo_moderate    ┆ house_diff      ┆ … ┆ womens health care                ┆ womens health cobr               ┆ 868 biddle st chesapeake city md                             ┆ chesapeake city 868 872 biddle st md                         │
│ US      ┆ S3  ┆ name_typo_moderate    ┆ house_diff      ┆ … ┆ orthopedic keystone center        ┆ orthopedic kyeseaotoe center     ┆ unit 102 chandler az 472 erie st                             ┆ 472 erie st 102 chandler az                                  │
│ US      ┆ S3  ┆ name_typo_moderate    ┆ cand_no_house   ┆ … ┆ feinberg trust                    ┆ feinberg tlbut                   ┆ 9902 oak leaf dr baytown tx                                  ┆ oak leaf dr baytown tx                                       │
└─────────┴─────┴───────────────────────┴─────────────────┴───┴───────────────────────────────────┴──────────────────────────────────┴──────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────┘
```

## Cap analysis: do misses share an UNCAPPED baseline key? (min block size of shared keys)
| country | no shared key | ≤300 | 300–1k | 1k–5k | >5k | shared-key share |
|---|---|---|---|---|---|---|
| US (21,064) | 8,151 | 6,315 | 3,061 | 1,670 | 1,867 | **61%** (mostly C,N1,N2 = full generic name) |
| India (21,392) | 14,023 | 3,987 | 1,509 | 963 | 910 | 34% (A and name keys) |

India no-shared-key misses: 8,276 native script, 1,886 moderate typo, 1,777 near-identical, 1,041 heavy diff.

## Conclusions → fixes
1. **Generic/shared names (US: ~8.4k near-identical misses are capped name keys).** About half of S1 names are shared with another S1, so name-only retrieval saturates.
   - Fix: a name+address TF-IDF view (weighted concat of the name and address vectors), so identical names are ranked by address.
   - Name-only TF-IDF top-k will help less here. It will mainly catch typos and extra words.
2. **Native script (India 9.7k misses, 3.2 recall pts).** anyascii output is lossy ("gret phaumdesn" vs "great foundation", "praivrr limirrd").
   - Fix: char 3-grams partially; a learned token map; a dense encoder as the fallback.
3. **Typos, heavy diffs, extra words (~11k).** Fix: char-3gram TF-IDF on the name, plus reverse retrieval.
4. **Domain/handle (2.4k).** The compact key is capped (C cap 60) or the prefix differs ("sciencesprivate"). Fix: char n-grams on n_compact are covered by char_wb on the core.
5. **House format:** zero-pad differences for 3.9k misses (`a_house.lstrip('0')` in key A).
