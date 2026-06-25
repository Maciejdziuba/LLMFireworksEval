"""Fireworks hosted evaluator entrypoint."""

from typing import Any

import pytest

from eval_protocol import (
    DynamicDataLoader,
    EvaluationRow,
    SingleTurnRolloutProcessor,
    evaluation_test,
    load_jsonl,
)

from llm_judge import aha_judge
from utils import assistant_to_ground_truth


JUDGE_MODEL = "kimi-k2-instruct-0905"


def dataset_adapter(rows: list[dict[str, Any]]) -> list[EvaluationRow]:
    """Convert uploaded JSONL rows and move the baseline answer to ground_truth."""
    return assistant_to_ground_truth([EvaluationRow(**row) for row in rows])


def uploaded_jsonl_rows() -> list[EvaluationRow]:
    """Load rows when EP_JSONL_PATH is present; otherwise keep collection side-effect free."""
    import os

    dataset_path = os.getenv("EP_JSONL_PATH")
    if not dataset_path:
        return []
    return [EvaluationRow(**row) for row in load_jsonl(dataset_path)]


@pytest.mark.parametrize(
    "completion_params",
    [
        {
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
            "max_tokens": 131000,
            "extra_body": {"reasoning_effort": "medium"},
        }
    ],
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[uploaded_jsonl_rows],
        preprocess_fn=assistant_to_ground_truth,
    ),
    dataset_adapter=dataset_adapter,
    rollout_processor=SingleTurnRolloutProcessor(),
    max_concurrent_evaluations=16,
)
async def test_llm_judge(row: EvaluationRow) -> EvaluationRow:
    return await aha_judge(row, judge_name=JUDGE_MODEL)
