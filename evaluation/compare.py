"""Compare paired benchmark JSON reports without inventing missing measurements."""
import argparse
import json
import statistics
from pathlib import Path

def compare(baseline, optimized):
    def indexed(report):
        rows = report["results"]
        result = {str(r["case"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate case identifiers")
        return result
    a, b = indexed(baseline), indexed(optimized)
    if a.keys() != b.keys():
        raise ValueError("Reports must cover identical case identifiers")
    pairs = [(a[k], b[k]) for k in a if "error" not in a[k] and "error" not in b[k]]
    output = {"attempted": len(a), "paired_successes": len(pairs),
              "baseline_failures": sum("error" in r for r in a.values()),
              "optimized_failures": sum("error" in r for r in b.values()), "metrics": {}}
    for key in ("precision", "recall", "hit", "latency_s", "retrieval_ms", "total_ms",
                "chat_input_tokens", "chat_output_tokens"):
        def val(r):
            return r.get(key, r.get("metrics", {}).get(key))
        values = [(val(x), val(y)) for x,y in pairs if val(x) is not None and val(y) is not None]
        if not values:
            continue
        old = statistics.mean(x for x,y in values)
        new = statistics.mean(y for x,y in values)
        output["metrics"][key] = {"paired_cases": len(values), "baseline_mean": old,
            "optimized_mean": new, "absolute_change": new-old,
            "relative_change_percent": 100*(new-old)/old if old else None}
    return output

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--optimized", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = compare(json.loads(Path(a.baseline).read_text()), json.loads(Path(a.optimized).read_text()))
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
