import csv, random, sys
random.seed(0)
D="student_resource/dataset/train/"
N=int(sys.argv[1]) if len(sys.argv)>1 else 3000
gt=[]
with open(D+"train_ground_truth.tsv") as f:
    next(f)
    for i,l in enumerate(f):
        if random.random()<N/2206821: gt.append(l.rstrip("\n").split("\t"))
need={}
for s1,m in gt:
    need[s1]=1
    for x in (m.split(",") if m else []): need[x]=1
recs={}
for fn in ["train_source1.tsv","train_source2.tsv","train_source3.tsv"]:
    with open(D+fn) as f:
        next(f)
        for l in f:
            k=l.split("\t",1)[0]
            if k in need: recs[k]=l.rstrip("\n")
with open("work/sample_clusters.tsv","w") as o:
    for s1,m in gt:
        o.write("### "+s1+"\n"+recs.get(s1,"?")+"\n")
        for x in (m.split(",") if m else []): o.write("   "+recs.get(x,"MISSING "+x)+"\n")
print(len(gt))
