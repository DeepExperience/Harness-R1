# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DAPO GenRM Reward Model Implementation.

This module implements a reward model using GenRM (Generative Reward Model) for
DAPO-style question answering tasks.
"""

from typing import List

from relax.utils.genrm_client import get_genrm_client
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# GenRM prompt template for DAPO
DAPO_GENRM_PROMPT_TEMPLATE = """Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just output Judgement and don't output anything else.

[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""


# Default truncation length for model predictions without explicit "Answer:" marker
GENRM_DEFAULT_TRUNCATION_LEN = 300


def _format_messages(question: str, ground_truth: str, predict_str: str) -> List[dict]:
    """Format DAPO prompt as OpenAI-style messages.

    Args:
        question: The question being evaluated
        ground_truth: The ground truth answer
        predict_str: The model's prediction to evaluate

    Returns:
        List of message dicts for GenRM service
    """

    if "Answer:" in predict_str:
        predict_str = predict_str.split("Answer:")[-1]
    else:
        predict_str = predict_str[-GENRM_DEFAULT_TRUNCATION_LEN:]

    prompt = DAPO_GENRM_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        predict_str=predict_str,
    )

    return [{"role": "user", "content": prompt}]


async def async_compute_score_genrm(args, sample) -> dict:
    """Compute reward score using GenRM service (async).

    This function is called by the rollout process to evaluate responses
    using the GenRM service. It must be async because it is invoked from
    the async ``async_rm`` dispatch in ``rm_hub/__init__.py``.

    Args:
        args: Argument namespace containing configuration
        sample: Sample object containing:
            - metadata: dict with 'question' and 'label' (ground truth)
            - response: str model response text

    Returns:
        Dictionary with keys:
            - score: float reward score (0 or 1)
            - acc: int accuracy (0 or 1)
            - pred: str prediction result
    """
    try:
        # Get singleton GenRM client
        genrm_client = get_genrm_client()

        # Extract data from sample
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        question = metadata.get("question", sample.prompt if hasattr(sample, "prompt") else "")
        ground_truth = metadata.get("label", sample.label if hasattr(sample, "label") else "")
        predict_str = sample.response

        # Format messages for GenRM service
        messages = _format_messages(question, ground_truth, predict_str)

        # Call GenRM service (async)
        # generate() returns the raw response string directly
        response = await genrm_client.generate(messages)

        # Parse result - response is now a string directly from generate()
        prediction = response.strip()

        # Extract judgement — use strict equality to avoid false positives
        # (e.g., "10" or "1 because..." should not match)
        if prediction == "1":
            score = 1.0
            acc = 1
        else:
            score = 0.0
            acc = 0

        return {
            "score": score,
            "acc": acc,
            "pred": prediction,
        }

    except Exception as e:
        logger.error(f"GenRM async_compute_score_genrm failed: {e}")
        raise
