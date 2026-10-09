"""Regenerate report figures from preserved experiments; never retrain a policy.

Run from any directory with the repository's Python environment.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.dpi": 240, "font.family": "DejaVu Sans"})
BLUE, ORANGE, GREEN = "#245b87", "#d47725", "#29836b"
sources = set()


def read_json(relative):
    sources.add(relative)
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative):
    sources.add(relative)
    return pd.read_csv(ROOT / relative)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.set(xlim=(0, 10.5), ylim=(0, 2.6))
    ax.axis("off")
    labels = ["Simulation\nMuJoCo / Gymnasium", "Commande humaine\n30 démonstrations",
              "Behavior cloning\nActor supervisé", "Transfert vers SAC\nActor + replay",
              "Évaluation\nValidation et test"]
    for i, label in enumerate(labels):
        x = i * 2.1 + .05
        ax.add_patch(FancyBboxPatch((x, .95), 1.85, .9,
                     boxstyle="round,pad=.04", facecolor="#edf3f7", edgecolor=BLUE))
        ax.text(x + .925, 1.4, label, ha="center", va="center", fontsize=10)
        if i < 4:
            ax.annotate("", xy=(x + 2.06, 1.4), xytext=(x + 1.89, 1.4),
                        arrowprops={"arrowstyle": "->", "color": BLUE})
    ax.text(5.25, .4, "Contrat commun : 38 observations · 4 actions incrémentales · succès par contact",
            ha="center", color=BLUE)
    save(fig, "pipeline")


def demonstrations():
    session = read_json("demonstrations/gamepad_30/session.json")
    split = read_json("runs/bc_gamepad/split.json")
    held = set(split["validation_episode_ids"])
    rows = session["completed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for validation, label, color in [(False, "Apprentissage (24)", BLUE),
                                      (True, "Validation (6)", ORANGE)]:
        selected = [r for r in rows if (r["episode"] in held) == validation]
        axes[0].scatter([r["spawn_position"][0] for r in selected],
                        [r["spawn_position"][1] for r in selected],
                        c=color, label=label, s=34)
    axes[0].set(xlim=(.3, .9), ylim=(-.3, .3), xlabel="Placement initial x (m)",
                ylabel="Placement initial y (m)", title="Couverture des placements")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].bar([r["episode"] for r in rows], [r["steps"] for r in rows],
                color=[ORANGE if r["episode"] in held else BLUE for r in rows])
    axes[1].set(xlabel="Épisode humain", ylabel="Nombre de transitions",
                title="Volume de données par épisode")
    fig.tight_layout()
    save(fig, "demonstrations")

    fig, ax = plt.subplots(figsize=(5, 3.3))
    for episode in (1, 8, 17, 30):
        relative = f"demonstrations/gamepad_30/episode_{episode:04d}.npz"
        sources.add(relative)
        with np.load(ROOT / relative, allow_pickle=False) as data:
            ax.plot(data["elapsed"], data["distances"], label=f"Épisode {episode}")
    ax.set(xlabel="Temps simulé (s)", ylabel="Distance site pince–centre cube (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    save(fig, "human_distances")


def bc():
    metrics = read_csv("runs/bc_gamepad/metrics.csv")
    summary = read_json("runs/bc_gamepad/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for col, label, color in [("train_mse", "Apprentissage", BLUE),
                               ("validation_mse", "Validation", ORANGE)]:
        axes[0].plot(metrics.epoch, metrics[col], label=label, color=color)
    axes[0].axvline(summary["best_epoch"], color=GREEN, ls="--", label="Checkpoint : époque 16")
    axes[0].set(yscale="log", xlabel="Époque (0 : initialisation)", ylabel="MSE des actions")
    axes[0].legend(fontsize=9)
    best = metrics.loc[metrics.epoch == summary["best_epoch"]].iloc[0]
    axes[1].bar(["J1", "J2", "J3", "Pince"],
                 [best[f"validation_mse_{i}"] for i in range(4)], color=BLUE)
    axes[1].set(yscale="log", ylabel="MSE de validation à l'époque 16")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save(fig, "bc_learning")
    rollout = read_csv("runs/bc_gamepad/best_validation_episodes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    labels = [str(s - 3000000) for s in rollout.seed]
    axes[0].bar(labels, rollout.elapsed, color=BLUE)
    axes[0].axhline(20, color=ORANGE, ls="--", label="Échéance : 20 s")
    axes[0].set(xlabel="Seed − 3 000 000", ylabel="Temps de contact simulé (s)", ylim=(0, 21))
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, rollout.final_distance * 100, color=GREEN)
    axes[1].set(xlabel="Seed − 3 000 000", ylabel="Distance finale entre centres (cm)")
    fig.tight_layout()
    save(fig, "bc_rollouts")


def sac():
    validation = read_csv("runs/sac_from_bc/validation.csv")
    stats = read_csv("runs/sac_from_bc/stats.csv")
    episodes = read_csv("runs/sac_from_bc/episodes.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.1), sharex=True)
    axes[0].plot(validation.global_step / 1000, validation.success_rate * 100,
                  "o-", color=BLUE, label="Validation déterministe (6 seeds)")
    axes[0].plot(stats.global_step / 1000, stats.success_rate_100 * 100,
                  color=ORANGE, label="Entraînement : ≤ 100 derniers épisodes")
    axes[0].set(ylabel="Réussite au contact (%)", ylim=(-5, 105))
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].annotate("Régression : 0/6", xy=(10, 0), xytext=(22, 22),
                     arrowprops={"arrowstyle": "->", "color": BLUE}, fontsize=10)
    axes[0].annotate("Rétablissement : 6/6", xy=(20, 100), xytext=(31, 65),
                     arrowprops={"arrowstyle": "->", "color": BLUE}, fontsize=10)
    axes[0].axvspan(0, 1, color=GREEN, alpha=.15)
    axes[1].plot(validation.global_step / 1000, validation.mean_reward,
                  "o-", color=BLUE, label="Validation")
    axes[1].plot(stats.global_step / 1000, stats.mean_reward_100,
                  color=ORANGE, label="Entraînement")
    axes[1].set(xlabel="Décisions d'interaction (milliers)", ylabel="Retour moyen")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "sac_progress")

    critics = read_csv("runs/sac_from_bc/critics.csv")
    actions = read_csv("runs/sac_from_bc/actions.csv")
    fig = plt.figure(figsize=(8, 5.2))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    axes[0].plot(stats.global_step / 1000, stats.critic_loss, color=BLUE)
    axes[0].set(yscale="log", ylabel="Somme des MSE des Critics")
    axes[1].plot(stats.global_step / 1000, stats.alpha, color=GREEN)
    axes[1].set(ylabel="Température α")
    for i, label in enumerate(["J1", "J2", "J3", "Pince"]):
        axes[2].plot(actions.global_step / 1000, 100 * actions[f"action_saturated_{i}"], label=label)
    axes[2].set(ylabel="Actions avec |a| ≥ 0,95 (%)", ylim=(0, 100))
    axes[2].legend(fontsize=9, ncol=4)
    for ax in axes:
        ax.set_xlabel("Décisions (milliers)")
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "sac_diagnostics")
    bc_rollout = read_csv("runs/bc_gamepad/best_validation_episodes.csv")
    audit = read_json("reports/sac_audit_metrics.json")
    session = read_json("demonstrations/gamepad_30/session.json")
    computed = {"demonstration_transitions": sum(r["steps"] for r in session["completed"]),
                "human_mean_simulated_seconds": float(np.mean([r["simulated_elapsed"] for r in session["completed"]])),
                "bc_mean_contact_seconds": float(bc_rollout.elapsed.mean()),
                "bc_mean_reward": float(bc_rollout.total_reward.mean()),
                "bc_mean_final_distance": float(bc_rollout.final_distance.mean()),
                "sac_last_validation": validation.iloc[-1].to_dict(),
                "sac_last_logged_episode": episodes.iloc[-1].to_dict(),
                "sac_last_stats": stats.iloc[-1].to_dict(),
                "historical_replay_below_30cm_percent": 100 * audit["replay"]["fractions_below"]["0.3"],
                "sac_last_critic_diagnostics": critics.iloc[-1].to_dict()}
    (OUT.parent / "computed_metrics.json").write_text(
        json.dumps(computed, ensure_ascii=False, indent=2), encoding="utf-8")


def scene():
    import mujoco
    relative = "models/robot_arm.xml"
    sources.add(relative)
    model = mujoco.MjModel.from_xml_path(str(ROOT / relative))
    data = mujoco.MjData(model)
    with np.load(ROOT / "demonstrations/gamepad_30/episode_0001.npz", allow_pickle=False) as demo:
        states = [demo["observations"][0], demo["next_observations"][-1]]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.35, 0, .55]
    camera.distance = 2.65
    camera.azimuth = 130
    camera.elevation = -22
    # Larger offscreen buffer changes rendering only, never the preserved XML.
    model.vis.global_.offwidth = 1200
    model.vis.global_.offheight = 900
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    with mujoco.Renderer(model, height=900, width=1200) as renderer:
        for ax, obs, label in zip(axes, states, ["État initial", "Contact final d'une démonstration humaine"]):
            data.qpos[:] = obs[:model.nq]
            data.qvel[:] = obs[model.nq:model.nq + model.nv]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            ax.imshow(renderer.render())
            ax.set_title(label, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    save(fig, "simulation")


def independent_comparison():
    base = "article/experiments/heldout_50"
    read_json(f"{base}/protocol.json")
    summary = read_json(f"{base}/summary.json")
    pairs = read_csv(f"{base}/paired_independent.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    x = pairs.seed - 9_000_000
    for prefix, label, color in [("bc", "BC", BLUE), ("sac", "BC + SAC", GREEN)]:
        success = pairs[f"{prefix}_success"].astype(bool)
        axes[0].plot(x, pairs[f"{prefix}_seconds"], "o-", ms=3, label=label, color=color)
        if (~success).any():
            axes[0].scatter(x[~success], pairs.loc[~success, f"{prefix}_seconds"],
                            marker="x", s=70, color=color, label=f"Échec {label}", zorder=5)
    axes[0].axhline(20, ls="--", color=ORANGE, label="Échéance")
    axes[0].set(ylabel="Durée de l'épisode (s)", ylim=(0, 21))
    axes[0].legend(fontsize=9, ncol=3, loc="upper right")
    axes[1].bar(x, pairs.delta_return, color=[GREEN if d >= 0 else ORANGE for d in pairs.delta_return])
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(xlabel="Seed − 9 000 000", ylabel="Écart de retour : SAC − BC")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save(fig, "independent_comparison")

    # Preserve per-seed validation timings in a publication-ready table.
    validation = read_csv(f"{base}/paired_validation.csv")
    lines = []
    for row in validation.itertuples():
        lines.append(f"{row.seed-3000000} & {row.bc_seconds:.3f} & {row.sac_seconds:.3f} & {row.delta_return:.2f} \\\\")
    table = (r"\begin{tabular}{rrrr}" + "\n" + r"\toprule" + "\n"
             + r"Seed$-3\,000\,000$ & BC (s) & SAC (s) & $\Delta G$ \\" + "\n"
             + r"\midrule" + "\n" + "\n".join(lines) + "\n"
             + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    (OUT.parent / "paired_validation_rows.tex").write_text(table, encoding="utf-8")
    metrics_path = OUT.parent / "computed_metrics.json"
    computed = json.loads(metrics_path.read_text(encoding="utf-8"))
    computed["paired_evaluation"] = summary
    metrics_path.write_text(json.dumps(computed, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    pipeline()
    demonstrations()
    bc()
    sac()
    scene()
    independent_comparison()
    manifest = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(sources)}
    (OUT.parent / "figure_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(list(OUT.glob('*.pdf')))} figures in {OUT}")
