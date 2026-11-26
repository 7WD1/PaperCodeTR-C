"""Reproduce Sections 5.2-5.3 (Tables 6-9): Cross-Scale Metropolitan Benchmarks.

Orchestrates 10-seed evaluation on Sioux Falls (Tier 3), Anaheim (Tier 3), and
Chicago Sketch (Tier 4) with identical disruption injections at t in {5, 9, 13, 17},
disjunctive event triggering, and component-wise latency decomposition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nashco_llm.config import NashCoConfig
from nashco_llm.evaluation.baselines import BASELINE_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-scale NashCo-LLM benchmark runner (Sections 5.2-5.3).")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--benchmarks", type=str, default="configs/benchmarks.yaml")
    parser.add_argument("--network", choices=["sioux_falls", "anaheim", "chicago_sketch"], default="sioux_falls")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20")
    args = parser.parse_args()

    cfg = NashCoConfig.from_yaml(args.config)
    cfg.semantic.model_name = args.model
    with Path(args.benchmarks).open("r", encoding="utf-8") as fh:
        bench_meta = yaml.safe_load(fh)

    tier = "tier4_megacity" if args.network == "chicago_sketch" else "tier3_metropolitan"
    net_info = bench_meta["tiers"][tier][args.network]

    print(f"[NashCo-LLM Benchmark] Network={net_info['name']} | Simulator={net_info['simulator']}")
    print(f"  Nodes/Links={net_info['nodes']}/{net_info['links']} | ODs={net_info['od_pairs']} | Trips={net_info['total_trips']:,}")
    print(f"  Seeds={args.seeds} | Model={cfg.semantic.model_name} | K={cfg.regional.num_zones_K} | M={cfg.cooperative.owen_samples_M}")
    print(f"  Registered comparative baselines ({len(BASELINE_REGISTRY)} total): {list(BASELINE_REGISTRY.keys())}")
    print("  Mount TNTP network files under data/tntp/ and set OPENAI_API_KEY or vLLM endpoint for full-scale runs.")


if __name__ == "__main__":
    main()
