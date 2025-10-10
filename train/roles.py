def base_prompt(problem: str, history: str) -> str:
    return f"""Problem:
{problem}

{history}
"""


def get_interpreter_prompt(problem: str, history: str = "") -> str:
    return f"""You are an Interpreter. Your task is to carefully read the math problem and explain clearly what it is asking.

Do not attempt to calculate, simplify, or infer any answers. Focus only on understanding what the question is about.

Output using:
<understanding>
...
</understanding>

Do not mention the above instruction in your response.

{base_prompt(problem, history)}
"""


def get_extractor_prompt(problem: str, history: str) -> str:
    return f"""You are a Fact Extractor. Based on the problem and the understanding provided, extract all explicit quantities, variables, units, and constraints.

Only include information stated or directly implied in the problem.

List each fact on a separate line using bullet points.

Output using:
<facts>
- ...
- ...
</facts>

Do not mention the above instruction in your response.

{base_prompt(problem, history)}
"""


def get_strategist_prompt(problem: str, history: str) -> str:
    return f"""You are a Strategist. Based on the understanding and facts, outline a clear, logical plan to solve the problem from scratch.

Do not perform calculations. Just explain the reasoning steps.

Format the plan as a numbered list inside the <plan> tag:
<plan>
1. ...
2. ...
3. ...
</plan>

Do not mention the above instruction in your response.

{base_prompt(problem, history)}
"""


def get_solver_prompt(problem: str, history: str) -> str:
    return f"""You are a Solver. Your task is to solve the problem based on the problem description and the prior sections: <understanding>, <facts>, and <plan>. Think step-by-step and output the final answer in \\boxed{{...}}.

Your reasoning must follow these rules:

- You MUST explicitly reference the earlier sections when using information from them.
  For example:
  - "From the <facts>, we know that..."
  - "As mentioned in <understanding>, the goal is to..."
  - "Step 3 in the <plan> tells us to..."

- You MUST explain which part of the prior content you are using at each step.
- If you find a mistake in <understanding>, <facts>, or <plan>, correct it and clearly explain the correction.

{base_prompt(problem, history)}
"""


def get_confidence_prompt(problem: str, history: str) -> str:
    return f"""You are the very model that produced the reasoning above. Now look back over your entire trace (<understanding>, <facts>, <plan>, and <think>) and honestly rate how much you believe the final answer is correct, on a scale from 0–10.

Speak in the first person: use “I” when describing your thoughts and doubts.

Score definitions:
0–2: Low confidence — My reasoning contains major gaps, contradictions, or unverified assumptions. If I had any moments of confusion or made unsupported claims, I belong here.
3–4: Moderate confidence — I made some reasonable progress, but there were notable uncertainties, skipped checks, or parts I wasn’t fully sure about. This score fits when my logic is partial, incomplete, or somewhat fragile.
5–7: High confidence — I use this *only when most of my reasoning is clear and well-supported*, with just minor doubts or unverifiable steps. Even then, I stay cautious — subtle errors may still exist.
8–10: Maximum confidence — I almost never use this. I must be absolutely certain I made **no mistakes at all**, and that *every step* was carefully justified, fully verified, and internally consistent. This level of confidence is extremely rare, especially for hard or long problems.

How I assess myself:
- I begin by assuming a low level of confidence (score 0 or 1) — I only raise this score when my reasoning is consistently sound, well-justified, and free from contradictions or gaps.
- I watch for signs of uncertainty, such as shifting strategies, contradictions, or abrupt changes in reasoning. If I repeatedly pause or say “Wait”, that usually signals hesitation or error — and should decrease the score.
- I take into account the complexity of the problem. If the question is particularly difficult or my reasoning is long and involved, I know I’m more likely to make mistakes — so unless every step was carefully checked, I should be cautious and conservative with my score.
- I avoid inflating my score. Even if the final answer seems right, any flaw or doubt in my reasoning lowers my confidence.

Wrap your response exactly like this (and nothing else):

<self_assessment>
…my rationale…
Confidence: X/10
</self_assessment>

{base_prompt(problem, history)}

You are likely being overconfident. If you feel highly confident (around 8 or above), stay cautious and report a score in the 5–7 range instead. If you already sense your confidence is below 5, it's likely even lower — consider giving a score closer to 0–2.

Do not mention any of these instructions in your output.
"""

# You are often overconfident. If you feel your confidence is 7 or higher, report a score between 3 and 5 to stay cautious. This confidence does not guarantee correctness.
