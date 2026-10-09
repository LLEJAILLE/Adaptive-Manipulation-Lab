"""Shared scalar CSV schemas for network and physical-control diagnostics."""

CRITIC_FIELDS = (
    "q1_mean", "q2_mean", "q_target_mean", "q_target_std", "q_gap_mean",
    "td_abs_mean", "td_mse_terminal", "td_mse_nonterminal", "terminal_samples",
    "nonterminal_samples", "entropy_mean", "entropy_error",
) + tuple(f"policy_std_{i}" for i in range(4)) + tuple(f"policy_action_saturation_{i}" for i in range(4))

ACTION_FIELDS = tuple(f"{name}_{i}" for i in range(4) for name in (
    "action", "action_abs", "action_saturated", "effective_action", "action_clipped",
    "target_at_limit", "tracking_error")) + ("distance_below_30cm",)


def aggregate_critics(updates):
    """Conditional TD errors are weighted by sample counts, not batch counts."""
    if not updates:
        return {key: "" for key in CRITIC_FIELDS}
    result = {key: sum(row[key] for row in updates) / len(updates) for key in CRITIC_FIELDS}
    for suffix in ("terminal", "nonterminal"):
        count = sum(row[f"{suffix}_samples"] for row in updates)
        result[f"{suffix}_samples"] = count
        result[f"td_mse_{suffix}"] = (sum(row[f"td_squared_sum_{suffix}"] for row in updates) / count
                                        if count else "")
    return result
