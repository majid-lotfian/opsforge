"""Run a small reproducible retrieval benchmark against a deployed OpsForge API.

Supply independently labeled JSONL cases with title, description, relevant_issue_numbers.
Do not use verbatim issue descriptions as test queries. Optional bearer token via env.
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

def run(url, cases):
    rows = []
    for i, case in enumerate(cases, 1):
        expected = set(map(str, case["relevant_issue_numbers"]))
        if not expected:
            raise ValueError("Each benchmark case needs relevance labels")
        req = urllib.request.Request(url.rstrip("/") + "/api/analyze",
            data=json.dumps({k: case[k] for k in ("title", "description")}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        if os.getenv("OPSFORGE_EVAL_TOKEN"):
            req.add_header("Authorization", "Bearer " + os.environ["OPSFORGE_EVAL_TOKEN"])
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                result = json.load(response)
            found = {str(s["issue_number"]) for s in result.get("sources", [])}
            rows.append({"case": i, "precision": len(found & expected)/max(1,len(found)),
                         "recall": len(found & expected)/len(expected),
                         "hit": int(bool(found & expected)),
                         "latency_s": time.monotonic()-start,
                         "method": result.get("retrieval_method"),
                         "metrics": result.get("metrics", {})})
        except Exception as exc:
            rows.append({"case": i, "error": type(exc).__name__})
        print(json.dumps(rows[-1]), flush=True)
    valid = [r for r in rows if "error" not in r]
    output = {"cases": len(rows), "successful": len(valid),
              "mean": {k: statistics.mean(r[k] for r in valid) for k in
                       ("precision", "recall", "hit", "latency_s")} if valid else {},
              "results": rows}
    if valid:
        output["latency_seconds"] = {
            "median": statistics.median(r["latency_s"] for r in valid),
            "p95_nearest_rank": sorted(r["latency_s"] for r in valid)[(95 * len(valid) + 99) // 100 - 1],
        }
        output["tokens"] = {
            key: sum(int(r["metrics"].get(key) or 0) for r in valid)
            for key in ("chat_input_tokens", "chat_output_tokens")
        }
        output["retrieval_methods"] = {
            method: sum(r["method"] == method for r in valid)
            for method in sorted({r["method"] for r in valid if r["method"]})
        }
    return output

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--output", default="evaluation.json")
    a = p.parse_args()
    with open(a.cases) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if not cases:
        p.error("No labeled cases")
    report = run(a.url, cases)
    with open(a.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "results"}, indent=2))
