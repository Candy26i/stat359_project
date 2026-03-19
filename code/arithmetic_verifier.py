"""Verifier for arithmetic solutions used in GRPO training."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re


@dataclass
class RewardWeights:
    final_correct: float = 1.0
    finish: float = 0.05          # 出现 Final Result/ANSWER 的小奖励（防止烂尾）
    format_ok: float = 0.05       # 有 Step + 有 Final Result 的小奖励
    step_correct: float = 0.02    # 每个正确 step 的奖励
    step_wrong: float = -0.01     # 每个错误 step 的惩罚
    step_cap: float = 0.20        # step 总分截断，防止刷步数
    gold_align: float = 0.15      # 与 golden solution 每步一致时的对齐奖励


class ArithmeticVerifier:
    """Verifier for arithmetic solutions.

    Supports multiple reward modes for GRPO:
      - "binary": final-answer correctness only
      - "format_correct": reward only if (correct AND template matches)
      - "stepwise": final correctness + step-level shaping (+ small finish/format bonus)
    """

    def __init__(self, reward_mode: str = "binary", weights: Optional[RewardWeights] = None):
        self.reward_mode = reward_mode
        self.weights = weights or RewardWeights()

    # -------------------------
    # Final answer extraction
    # -------------------------
    def extract_final_result(self, generated_text: str) -> Optional[int]:
        """Extract final numeric result from generated text."""
        if not generated_text:
            return None

        # Explicit ERROR marker
        if re.search(r"(Final\s*Result|ANSWER)\s*:\s*ERROR\b", generated_text, flags=re.IGNORECASE):
            return None

        # Prefer explicit markers
        for pat in [
            r"Final\s*Result\s*:\s*([+-]?\s*\d+)",
            r"ANSWER\s*:\s*([+-]?\s*\d+)",
            r"Final\s*:\s*([+-]?\s*\d+)",
        ]:
            m = re.search(pat, generated_text, flags=re.IGNORECASE)
            if m:
                raw = m.group(1).replace(" ", "")
                try:
                    return int(raw)
                except ValueError:
                    return None

        # Optional fallback: last integer in the text
        # (helps reduce predicted=None; you can comment this out if you want stricter parsing)
        # ints = re.findall(r"([+-]?\s*\d+)", generated_text)
        # if ints:
        #     raw = ints[-1].replace(" ", "")
        #     try:
        #         return int(raw)
        #     except ValueError:
        #         return None

        return None

    # -------------------------
    # Format checks
    # -------------------------
    def _format_ok(self, text: str) -> bool:
        has_step = re.search(r"\bStep\s*\d+\s*:", text, flags=re.IGNORECASE) is not None
        has_final = re.search(r"\b(Final\s*Result|ANSWER)\s*:", text, flags=re.IGNORECASE) is not None
        return bool(has_step and has_final)

    # -------------------------
    # Step-wise scoring helpers
    # -------------------------
    _STEP_RE = re.compile(
        r"\bStep\s*(\d+)\s*:\s*([0-9\-\+\(\)\s]+)\s*=\s*([\-]?\s*\d+)\b",
        flags=re.IGNORECASE
    )

    def _safe_eval_int_expr(self, expr: str) -> Optional[int]:
        """Safely eval an integer arithmetic expr containing only digits, +/-, parentheses, spaces."""
        expr = expr.strip()
        if not expr:
            return None
        if re.fullmatch(r"[0-9\-\+\(\)\s]+", expr) is None:
            return None
        try:
            val = eval(expr, {"__builtins__": None}, {})
        except Exception:
            return None
        if isinstance(val, bool):
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, float) and abs(val - round(val)) < 1e-9:
            return int(round(val))
        return None

    def _score_steps(self, text: str) -> Tuple[float, int, int]:
        """Return (step_reward, n_correct, n_total)."""
        matches = list(self._STEP_RE.finditer(text))
        if not matches:
            return 0.0, 0, 0

        step_reward = 0.0
        correct = 0
        total = 0

        for m in matches:
            lhs_expr = m.group(2)
            rhs_str = m.group(3)
            rhs = int(rhs_str.replace(" ", ""))  # "- 13" -> "-13"
            lhs = self._safe_eval_int_expr(lhs_expr)

            total += 1
            if lhs is not None and lhs == rhs:
                correct += 1
                step_reward += self.weights.step_correct
            else:
                step_reward += self.weights.step_wrong

        # Clamp step reward so model can’t farm steps infinitely
        step_reward = max(-self.weights.step_cap, min(self.weights.step_cap, step_reward))
        return step_reward, correct, total

    def _extract_step_tuples(self, text: str) -> List[Tuple[str, int]]:
        """Extract (lhs_expr_normalized, result) for each step. Used for gold alignment."""
        out: List[Tuple[str, int]] = []
        for m in self._STEP_RE.finditer(text or ""):
            lhs = re.sub(r"\s+", " ", m.group(2).strip())
            try:
                rhs = int(m.group(3).replace(" ", ""))
            except (ValueError, TypeError):
                continue
            out.append((lhs, rhs))
        return out

    # -------------------------
    # Main reward
    # -------------------------
    def compute_reward(self, generated_text: str, ground_truth: int, gold_solution: Optional[str] = None) -> float:
        """Compute reward for generated response."""
        text = generated_text or ""

        # Final correctness
        result = self.extract_final_result(text)
        final_correct = (result is not None and result == ground_truth)
        r_final = self.weights.final_correct if final_correct else 0.0

        # Gold alignment: if each step matches golden solution, add reward
        r_align = 0.0
        if gold_solution and gold_solution.strip():
            gold_steps = self._extract_step_tuples(gold_solution)
            gen_steps = self._extract_step_tuples(text)
            if gold_steps:
                n_match = sum(
                    1 for i, g in enumerate(gold_steps)
                    if i < len(gen_steps) and gen_steps[i] == g
                )
                r_align = self.weights.gold_align * (n_match / len(gold_steps))
            elif not gen_steps and final_correct:
                r_align = self.weights.gold_align

        if self.reward_mode == "binary":
            return float(r_final + r_align)

        fmt_ok = self._format_ok(text)

        if self.reward_mode == "format_correct":
            # Setting 2: reward only if (correct AND template matches)
            return float(self.weights.final_correct if (final_correct and fmt_ok) else 0.0) + r_align

        if self.reward_mode == "stepwise":
            # Small bonus for explicitly finishing
            has_final_marker = re.search(r"\b(Final\s*Result|ANSWER)\s*:", text, flags=re.IGNORECASE) is not None
            r_finish = self.weights.finish if has_final_marker else 0.0

            # Optional small bonus for having the expected structure
            r_fmt = self.weights.format_ok if fmt_ok else 0.0

            # Step-wise shaping
            r_steps, _, _ = self._score_steps(text)

            return float(r_final + r_finish + r_fmt + r_steps + r_align)

        # Unknown mode: fallback to binary
        return float(r_final + r_align)