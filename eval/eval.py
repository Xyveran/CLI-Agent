"""
eval.py - Evaluation runner for the CLI agent.

Usage:
    python eval/eval.py                             # run all scenarios
    python eval/eval.py --scenario S3               # run one scenario
    python eval/eval.py --dry-run                   # validate scenario definitions
    python eval/eval.py --report eval_report.json

What it measures
-----------------------------------
- Task completion rate
- Tool calls made per run
- Manual steps saved vs agent steps
- Token usage per scenario
"""

import os
import sys
import json
import time
import argparse
from dataclasses import asdict


#
# Allow running from the project root: python eval/eval.py
#

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.scenarios import SCENARIOS, Scenario
from eval.report import print_report
from eval.runner import AgentRunner
from eval.evaluator import evaluate_scenario, ScenarioResult


#
# CLI entrypoint
#

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CLI agent")
    parser.add_argument(
        "--scenario", help="Run a single scenario by ID (e.g. S3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate scenario definitions without calling the API",
    )
    parser.add_argument(
        "--report",
        metavar="FILE",
        help="Save results as JSON to FILE (e.g. eval_report.json)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s.id == args.scenario]
        if not scenarios:
            print(f"Unknown scenario ID: {args.scenario}")
            sys.exit(1)

    runner = AgentRunner(verbose=args.verbose)
    results: list[ScenarioResult] = []

    for i, scenario in enumerate(scenarios):
        print(f"Running [{scenario.id}] {scenario.description} ...", end=" ", flush=True)
        result = evaluate_scenario(scenario, runner, dry_run=args.dry_run)
        results.append(result)
        print("PASS" if result.passed else "FAIL")

        if i < len(scenarios) - 1:
            time.sleep(1.0) # pacing between scenarios

    print_report(results)

    if args.report:
        with open(args.report, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()