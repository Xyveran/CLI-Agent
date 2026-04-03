"""
runner.py -> AgentRuner, _call_api, _call_and_validate

evaluator.py -> evaluate_scenario, ScenarioResult, _effort_reduction

report.py -> print_report

eval.py -> main() (and argparse)
"""

import time
from dataclasses import dataclass, asdict
from eval.scenarios import SCENARIOS, Scenario
from eval.runner import AgentRunner
from typing import Optional

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
            tool_calls_made=tool_calls,
            manual_steps=scenario.manual_steps,
            agent_steps=agent_steps,
            effort_reduction_pct=_effort_reduction(
                scenario.manual_steps, agent_steps),
            duration_seconds=round(elapsed, 2),
            prompt_tokens=result["prompt_tokens"],
            response_tokens=result["response_tokens"],
            final_output=output[:500],  # truncate for report
            failure_reason=failure_reason,
        )

def _effort_reduction(manual: int, agent: int) -> float:
    if manual <= 0:
        return 0.0
    return round( (1 - agent / manual) * 100, 1)