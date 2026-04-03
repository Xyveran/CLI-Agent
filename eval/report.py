from eval.evaluator import ScenarioResult


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
            print(f"        Failure     : {r.failure_reason}")

    print("\n" + "=" * 50 + "\n")