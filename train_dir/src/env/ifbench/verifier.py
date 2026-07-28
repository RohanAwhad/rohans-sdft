"""IFBench constraint verification wrapper.

Wraps the IFBench evaluation logic (loose mode) into a single function
that checks whether a response satisfies a set of constraints.
"""

from typing import Optional

from src.env.ifbench import instructions_registry


def verify_response(
    response: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict],
    prompt: str,
) -> tuple[bool, list[dict]]:
    """Check whether *response* satisfies every constraint.

    Uses IFBench "loose" evaluation: tries several text variants
    (strip markdown *, remove first/last lines) before declaring failure.

    Args:
        response: The API model's response text.
        instruction_id_list: e.g. ["count:word_count_range", "format:emoji"].
        kwargs_list: matching kwargs dicts for each instruction's build_description.
        prompt: The original user prompt (needed by some checkers).

    Returns:
        (all_pass, results) where results is a list of per-constraint dicts:
            [{"instruction_id": str, "passed": bool, "description": str}, ...]
    """
    if not response or not response.strip():
        return False, [
            {"instruction_id": iid, "passed": False, "description": "Empty response"}
            for iid in instruction_id_list
        ]

    # Build text variants (loose mode, same as evaluation_lib.py)
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    results: list[dict] = []
    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Filter out None-valued kwargs (same as evaluation_lib)
        kw = kwargs_list[index]
        clean_kwargs = {k: v for k, v in kw.items() if v is not None} if kw is not None else {}
        instruction.build_description(**clean_kwargs)

        # Some checkers need the prompt injected
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        description = instruction.build_description(**clean_kwargs)

        passed = False
        for variant in all_responses:
            if variant.strip() and instruction.check_following(variant):
                passed = True
                break

        results.append({
            "instruction_id": instruction_id,
            "passed": passed,
            "description": description,
        })

    all_pass = all(r["passed"] for r in results)
    return all_pass, results
