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
from utils.retry import with_backoff
from dataclasses import dataclass, asdict
from typing import Optional

#
# Allow running from the project root: python eval/eval.py
#

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.scenarios import SCENARIOS, Scenario

#
# Result dataclass
#

@dataclass
class ScenarioResult:
    scenario_id: str
    description: str
    passed: bool
    tool_calls_made: int
    manual_steps: int
    agent_steps: int        # always 1 (single CLI command)
    effort_reduction_pct: float
    duration_seconds: int
    prompt_tokens: int
    response_tokens: int
    final_output: str
    failure_reason: Optional[str] = None

#
# Agent runner (wraps main.generate_content agentic loop)
#

class AgentRunner:
    """Thin wrapper around the project's main.py agentic loop."""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._client = None
        self._available_functions = None
        self._system_prompt = None
    
    def _ensure_client(self):
        if self._client is not None:
            return
        
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Export it before running evals."
            )
    
        from google import genai
        from google.genai import types
        import prompts
        from functions.call_function import available_functions

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._available_functions = available_functions
        self._system_prompt = prompts.system_prompt

    def run(self, prompt: str) -> dict:
        """
        Run the agent on `prompt`.
        
        Returns a dict with keys:
            output (str), tool_calls (int), prompt_tokens (int), response_tokens (int)
        """
        self._ensure_client()

        types = self._types
        client = self._client

        messages = [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]

        total_tool_calls = 0
        total_prompt_tokens = 0
        total_response_tokens = 0
        final_output = ""

        for _ in range(10):
            response = _call_api(
                client,
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    tools=[self._available_functions],
                    system_instruction=self._system_prompt,
                ),
                contents=messages,
            )

            # response = client.models.generate_content(
            #     model="gemini-2.5-flash",
            #     config=types.GenerateContentConfig(
            #         tools=[self._available_functions],
            #         system_instruction=self._system_prompt,
            #     ),
            #     contents=messages,
            # )

            if response.usage_metadata:
                total_prompt_tokens += response.usage_metadata.prompt_token_count or 0

                total_response_tokens += (
                    response.usage_metadata.candidates_token_count or 0
                )
            
            if not response.function_calls:
                final_output = response.text or ""
                break

            # Process function calls
            from functions.call_function import call_function

            function_results = []
            for fc in response.function_calls:
                total_tool_calls += 1
                fr = call_function(fc, verbose=self.verbose)
                function_results.append(fr)

            messages.append(
                types.Content(role="user", parts=function_results)
            )
        
        return {
            "output": final_output,
            "tool_calls": total_tool_calls,
            "prompt_tokens": total_prompt_tokens,
            "response_tokens": total_response_tokens,
        }
    
@with_backoff(max_retries=5, base_delay=2.0) # slightly longer base for eval runs
def _call_api(client, model, config, contents):
    return client.models.generate_content(
        model=model, config=config, contents=contents,
    )

#
# Evaluator
#

def evaluate_scenario(
    scenario: Scenario,
    runner: AgentRunner,
    dry_run: bool = False,
) -> ScenarioResult:
    """Run a single scenario and return a ScenarioResult."""

    agent_steps = 1     # agent is always invoked with one command

    if dry_run:
        return ScenarioResult(
            scenario_id=scenario.id,
            description=scenario.description,
            passed=True,
            tool_calls_made=0,
            manual_steps=scenario.manual_steps,
            agent_steps=agent_steps,
            effort_reduction_pct=_effort_reduction(
                scenario.manual_steps, agent_steps),
            duration_seconds=0.0,
            prompt_tokens=0,
            response_tokens=0,
            final_output="[dry-run]",
        )
    
    start = time.time()
    try:
        result = runner.run(scenario.prompt)
    except Exception as e:
        elapsed = time.time() - start
        return ScenarioResult(
            scenario_id=scenario.id,
            description=scenario.description,
            passed=False,
            tool_calls_made=0,
            manual_steps=scenario.manual_steps,
            agent_steps=agent_steps,
            effort_reduction_pct=_effort_reduction(
                scenario.manual_steps, agent_steps),
            duration_seconds=round(elapsed, 2),
            prompt_tokens=0,
            response_tokens=0,
            final_output="",
            failure_reason=str(e),
        )
    
    elapsed = time.time() - start

    output = result["output"]
    tool_calls = result["tool_calls"]

    # Determine pass/fail
    passed = True
    failure_reason = None

    if scenario.validator:
        if not scenario.validator(output):
            passed = False
            failure_reason = "Validator returned False for agent output"

    if tool_calls < scenario.expected_min_tool_calls:
        passed = False
        failure_reason = (
            f"Expected >{scenario.expected_min_tool_calls} tool calls, got {tool_calls}"
        )

    return ScenarioResult(
            scenario_id=scenario.id,
            description=scenario.description,
            passed=passed,
            tool_calls_made=0,
            manual_steps=scenario.manual_steps,
            agent_steps=agent_steps,
            effort_reduction_pct=_effort_reduction(
                scenario.manual_steps, agent_steps),
            duration_seconds=round(elapsed, 2),
            prompt_tokens=result["prompt_tokens"],
            response_tokens=result["response_tokens"],
            final_output=output[:500], # truncate for report
            failure_reason=failure_reason,
        )

def _effort_reduction(manual: int, agent: int) -> float:
    if manual <= 0:
        return 0.0
    return round( (1 - agent / manual) * 100, 1)

#
# Report printer
#

def print_report(results: list[ScenarioResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    completion_rate = round(passed / total * 100, 1) if total else 0

    avg_reduction = (
        sum(r.effort_reduction_pct for r in results) / total if total else 0
    )

    print("\n" + "=" * 50)
    print("  CLI AGENT — EVALUATION REPORT")
    print("=" * 50)
    print(f"  Scenarios run   : {total}")
    print(f"  Passed          : {passed}  |  Failed: {total - passed}")
    print(f"  Completion rate : {completion_rate}%")
    print(f"  Avg effort saved: {avg_reduction:.1f}%")
    print("=" * 50)

    for r in results:
        status = " PASS" if r.passed else " FAIL"
        print(
            f"\n [{r.scenario_id}] {status} - {r.description}"
        )
        print(f"        Tool cals   : {r.tool_calls_made}")
        print(
            f"      Effort saved : {r.manual_steps} manual steps -> "
            f"{r.agent_steps} agent step ({r.effort_reduction_pct}% reduction)"
        )
        print(f"        Duraton     : {r.duration_seconds}s")
        if r.prompt_tokens:
            print(
                f"      Tokens      : {r.prompt_tokens} prompt /"
                f"{r.response_tokens} response"
            )
        if r.failure_reason:
            print(f"        Failure     : {r.failure_reason} /")

    print("\n" + "=" * 50 + "\n")

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
        actions="store_true",
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