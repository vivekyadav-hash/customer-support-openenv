"""
Inference Script — Customer Support Triage OpenEnv
===================================================
Follows mandatory stdout format:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────
#  ENV VARIABLES
# ─────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "customer-support-triage"
SUCCESS_SCORE_THRESHOLD = 0.5

TASK_LEVELS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert customer support agent.
    You will receive a customer support ticket and must respond with:
    1. priority: one of [low, medium, high, critical]
    2. department: one of [billing, technical, general]
    3. response_draft: a helpful, empathetic reply to the customer (minimum 20 words)

    Rules:
    - Use "critical" only for outages, data loss, or enterprise emergencies
    - Use "high" for billing issues, login failures, or urgent complaints
    - Use "medium" for general questions or minor issues
    - Use "low" for feedback or non-urgent requests
    - Route to "technical" for login, password, platform, outage issues
    - Route to "billing" for invoice, charge, refund, payment issues
    - Route to "general" for everything else

    You MUST respond in this exact format (no extra text):
    PRIORITY: <value>
    DEPARTMENT: <value>
    RESPONSE: <your draft reply>
""").strip()


# ─────────────────────────────────────────
#  LOGGING FUNCTIONS (mandatory format)
# ─────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ")[:100]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────
#  ENVIRONMENT CALLS
# ─────────────────────────────────────────
def env_reset(task_level: str) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_level": task_level}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(priority: str, department: str, response_draft: str) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/step", json={
        "priority": priority,
        "department": department,
        "response_draft": response_draft,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────
#  LLM CALL
# ─────────────────────────────────────────
def get_agent_action(client: OpenAI, ticket_text: str, task_level: str) -> dict:
    user_prompt = f"Task level: {task_level}\n\nTicket:\n{ticket_text}\n\nRespond now."
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(text)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {
            "priority": "medium",
            "department": "general",
            "response_draft": "Thank you for contacting us. We will look into your issue and get back to you shortly.",
        }


def parse_llm_response(text: str) -> dict:
    """Parse LLM response into priority, department, response_draft."""
    lines = text.strip().split("\n")
    result = {
        "priority": "medium",
        "department": "general",
        "response_draft": "Thank you for contacting us. We will resolve your issue as soon as possible.",
    }
    response_lines = []
    in_response = False

    for line in lines:
        line = line.strip()
        if line.upper().startswith("PRIORITY:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ["low", "medium", "high", "critical"]:
                result["priority"] = val
        elif line.upper().startswith("DEPARTMENT:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ["billing", "technical", "general"]:
                result["department"] = val
        elif line.upper().startswith("RESPONSE:"):
            val = line.split(":", 1)[1].strip()
            response_lines.append(val)
            in_response = True
        elif in_response and line:
            response_lines.append(line)

    if response_lines:
        result["response_draft"] = " ".join(response_lines)

    return result


# ─────────────────────────────────────────
#  MAIN — runs all 3 tasks
# ─────────────────────────────────────────
def main() -> None:
    if not API_KEY:
        print("[ERROR] No API key found. Set HF_TOKEN or API_KEY environment variable.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_rewards = []
    all_steps = 0
    all_success = True

    for task_level in TASK_LEVELS:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        error_msg = None

        log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

        try:
            # Reset environment
            obs = env_reset(task_level)
            ticket_text = obs["ticket_text"]

            # Get LLM action
            action = get_agent_action(client, ticket_text, task_level)

            # Step environment
            result = env_step(
                priority=action["priority"],
                department=action["department"],
                response_draft=action["response_draft"],
            )

            reward = result.get("reward") or 0.0
            done = result.get("done", True)
            steps_taken = 1
            rewards.append(reward)

            action_str = f"priority={action['priority']} dept={action['department']}"
            log_step(step=1, action=action_str, reward=reward, done=done, error=None)

            score = round(reward, 3)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            error_msg = str(exc)
            print(f"[DEBUG] Task {task_level} failed: {error_msg}", flush=True)
            rewards = [0.0]
            steps_taken = 1
            score = 0.0
            success = False
            log_step(step=1, action="error", reward=0.0, done=True, error=error_msg)

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        all_rewards.extend(rewards)
        all_steps += steps_taken
        if not success:
            all_success = False

    # Final summary
    avg_score = round(sum(all_rewards) / len(all_rewards), 3) if all_rewards else 0.0
    print(f"\n[SUMMARY] tasks={len(TASK_LEVELS)} avg_score={avg_score:.3f} total_steps={all_steps}", flush=True)


if __name__ == "__main__":
    main()
