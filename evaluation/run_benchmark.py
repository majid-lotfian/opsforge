"""Resumable authenticated benchmark runner; standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request
import urllib.error

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",required=True)
    p.add_argument("--cases",required=True)
    p.add_argument("--split",choices=["development","heldout"],required=True)
    p.add_argument("--variant",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    raw=Path(a.cases).read_bytes()
    cases=[c for line in raw.splitlines() if line.strip() for c in [json.loads(line)] if c.get("split")==a.split]
    ids=[str(c["id"]) for c in cases]
    if len(cases)!={"development":100,"heldout":400}[a.split] or len(set(ids))!=len(ids):
        p.error("Expected complete split with unique case IDs")
    if any(not c.get("relevant_issue_numbers") for c in cases):
        p.error("Missing reference issues")
    meta={"dataset_sha256":hashlib.sha256(raw).hexdigest(),"split":a.split,"variant":a.variant,"url":a.url}
    print(json.dumps({**meta,"cases":len(cases)}),flush=True)
    if not a.execute: return
    token=os.getenv("OPSFORGE_EVAL_TOKEN")
    if not token: p.error("Set OPSFORGE_EVAL_TOKEN securely before execution")
    path=Path(a.output)
    checkpoint=path.with_suffix(".jsonl")
    manifest=path.with_suffix(".manifest.json")
    if manifest.exists() and json.loads(manifest.read_text())!=meta:
        p.error("Checkpoint metadata mismatch")
    manifest.write_text(json.dumps(meta))
    rows=[json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()] if checkpoint.exists() else []
    done={r["case"] for r in rows}
    for case in cases:
        cid=str(case["id"])
        if cid in done: continue
        req=urllib.request.Request(a.url.rstrip("/")+"/api/analyze",data=json.dumps({k:case[k] for k in ("title","description")}).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+token},method="POST")
        start=time.monotonic()
        try:
            with urllib.request.urlopen(req,timeout=240) as resp:
                if "application/json" not in resp.headers.get("Content-Type",""): raise RuntimeError("Non-JSON response; check authentication")
                result=json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (401,403,429): raise SystemExit(f"Stopped on HTTP {e.code}; resume after resolving")
            row={"case":cid,"error":f"HTTP {e.code}"}
        except RuntimeError: raise
        except Exception as e:
            row={"case":cid,"error":type(e).__name__}
        else:
            expected=set(map(str,case["relevant_issue_numbers"]))
            found={str(s["issue_number"]) for s in result.get("sources",[])}
            row={"case":cid,"precision":len(expected&found)/max(1,len(found)),
                 "recall":len(expected&found)/len(expected),"hit":int(bool(expected&found)),
                 "latency_s":time.monotonic()-start,"metrics":result.get("metrics",{}),
                 "method":result.get("retrieval_method"),"review_status":result.get("review_status"),
                 "expected":sorted(expected),"cited":sorted(found),"investigation":result}
        with checkpoint.open("a") as f: f.write(json.dumps(row)+"\n")
        rows.append(row)
        print(f"{len(rows)}/{len(cases)} completed: {cid}",flush=True)
    valid=[r for r in rows if "error" not in r]
    report={**meta,"cases":len(rows),"successful":len(valid),"results":rows,
            "mean":{k:statistics.mean(r[k] for r in valid) for k in ("precision","recall","hit","latency_s")} if valid else {}}
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2))

if __name__=="__main__":
    main()
