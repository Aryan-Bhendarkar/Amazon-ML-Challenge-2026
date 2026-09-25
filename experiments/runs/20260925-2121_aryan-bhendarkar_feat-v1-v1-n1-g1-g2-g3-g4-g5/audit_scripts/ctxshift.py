import os; os.environ["POLARS_MAX_THREADS"]="2"
import sys; sys.path.insert(0,"src")
import polars as pl
from ber import ctx_features as cf
for split in ("train","test"):
    ctx = cf.SplitContext.build(split)
    s1 = (pl.scan_parquet(cf._norm_file(split,1)).select("country","n_core","a_full","a_street","a_house")
            .with_columns(cf.key_exprs()).collect())
    pool_n = pl.concat([pl.scan_parquet(cf._norm_file(split,s)).select("country").collect() for s in (2,3)]).group_by("country").len("pool")
    for tbl,key,nm in ((ctx.s1_name,"n_key","s1_name"),(ctx.s1_addr,"a_key","s1_addr"),(ctx.s1_hs,"hs_key","s1_hs"),(ctx.pool_name,"n_key","pool_name"),(ctx.pool_addr,"a_key","pool_addr")):
        s1 = s1.join(tbl.rename({"cnt":nm}), on=["country",key], how="left").with_columns(pl.col(nm).fill_null(0).clip(0,20))
    agg = s1.group_by("country").agg(pl.len().alias("n_s1"),
        (pl.col("s1_name")==1).mean().alias("name_uniq"), (pl.col("s1_name")>=5).mean().alias("name>=5"),
        (pl.col("s1_addr")==1).mean().alias("addr_uniq"), (pl.col("s1_addr")>=5).mean().alias("addr>=5"),
        (pl.col("s1_hs")==1).mean().alias("hs_uniq"), (pl.col("hs_key")=="").mean().alias("hs_empty"),
        pl.col("pool_name").mean().alias("pool_name_mean"), (pl.col("pool_name")==0).mean().alias("pool_name0"),
        pl.col("pool_addr").mean().alias("pool_addr_mean"), (pl.col("pool_addr")==0).mean().alias("pool_addr0"),
    ).join(pool_n, on="country").with_columns((pl.col("pool")/pl.col("n_s1")).alias("pool_per_s1")).sort("country")
    idf = ctx.idf.join(ctx.max_idf(), on="country").group_by("country").agg(pl.col("idf").median().alias("idf_med"), pl.col("idf_max").first())
    pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
    print(split); print(agg.join(idf, on="country"))
