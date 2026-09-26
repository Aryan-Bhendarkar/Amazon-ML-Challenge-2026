import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, pyarrow.parquet as pq, time
S="/home/ubuntu/amlc/submissions/files/20260926-0953_aryan-bhendarkar"
A="/home/ubuntu/amlc/artifacts/20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3"
t0=time.time()
raw=open(f"{S}/matching_results.tsv","rb").read(); print("CR bytes in matching:",raw.count(b"\r"), "spaces:", raw.count(b" ")); del raw
m=pl.read_csv(f"{S}/matching_results.tsv",separator="\t",schema_overrides={"source1_entity_id":pl.Utf8,"matched_entity_ids":pl.Utf8},quote_char=None,missing_utf8_is_empty_string=True)
mx=m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids")!="").rename({"source1_entity_id":"s1_id","matched_entity_ids":"match_id"})
print("tsv rows",m.height,"pairs",mx.height,"dup match_id",mx.height-mx["match_id"].n_unique(), "dup s1 rows", m.height-m["source1_entity_id"].n_unique())
pm=pl.read_parquet(f"{A}/test_matches.parquet")
print("parquet matches",pm.height,"anti tsv->pq",mx.join(pm,on=["s1_id","match_id"],how="anti").height,"anti pq->tsv",pm.join(mx,on=["s1_id","match_id"],how="anti").height)
# candidates parquet: membership + hash-sum fingerprint
pf=pq.ParquetFile("/home/ubuntu/amlc/data/cands/v1_n2/test.parquet")
insum=0; hsum=0; n=0
for rg in range(pf.num_row_groups):
    c=pl.from_arrow(pf.read_row_group(rg,columns=["s1_id","cand_id"]))
    n+=c.height; hsum=(hsum+int(c.select(pl.concat_str("s1_id","cand_id",separator="|").hash(seed=3).cast(pl.UInt64).sum()).item()))%(1<<64)
    insum+=mx.join(c,left_on=["s1_id","match_id"],right_on=["s1_id","cand_id"],how="semi").height
print("parquet cand rows",n,"matches in cands",insum,"of",mx.height,"hash",hsum)
# candidate TSV fingerprint, streamed
hs=0; nc=0; rows=0; cr=0
with open(f"{S}/candidate_pairs.tsv","rb") as f:
    hdr=f.readline(); print("cand header",hdr)
    buf=[]
    def flush(buf):
        global hs,nc
        d=pl.DataFrame({"l":buf}).with_columns(pl.col("l").str.split_exact("\t",1)).unnest("l").rename({"field_0":"s1_id","field_1":"ids"})
        e=d.with_columns(pl.col("ids").str.split(",")).explode("ids").filter(pl.col("ids")!="")
        nc+=e.height; hs=(hs+int(e.select(pl.concat_str("s1_id","ids",separator="|").hash(seed=3).cast(pl.UInt64).sum()).item()))%(1<<64)
    for line in f:
        if line.endswith(b"\r\n"): cr+=1
        buf.append(line.rstrip(b"\n").decode()); rows+=1
        if len(buf)>=100000: flush(buf); buf=[]
    if buf: flush(buf)
print("cand tsv rows",rows,"ids",nc,"CRLF lines",cr,"hash",hs,"hash equal",hs==hsum, f"{time.time()-t0:.0f}s")
