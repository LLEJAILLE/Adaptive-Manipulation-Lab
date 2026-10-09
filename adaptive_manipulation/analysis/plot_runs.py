"""Plot BC or SAC progress from the CSV files in a run folder."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load(run_dir, name):
    path = run_dir / f"{name}.csv"
    if not path.exists():
        print(f"Warning: {path} not found, skipping")
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m adaptive_manipulation plot', description=__doc__)
    ap.add_argument("run_dir", help="path to a folder inside runs/")
    ap.add_argument("--save", help="save figure to this path")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--window", type=int, default=20, help="moving average window for episodes")
    args = ap.parse_args(argv)
    if args.window < 1:
        ap.error('--window must be positive')

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    if (run_dir / 'metrics.csv').exists():
        plot_bc(run_dir, args)
        return

    ep, st, va = (load(run_dir, n) for n in ("episodes", "stats", "validation"))
    # Old runs remain plottable; only add this figure when new diagnostics exist.
    critics = load(run_dir, "critics") if (run_dir / "critics.csv").exists() else None
    actions = load(run_dir, "actions") if (run_dir / "actions.csv").exists() else None

    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
    fig.suptitle(f"Training progress: {run_dir.name}")
    x = "global_step"

    def draw(ax, df, cols, title):
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if df is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        for c in cols:
            ax.plot(df[x], df[c], marker="o", ms=3, label=c)
        if len(cols) > 1:
            ax.legend()

    def draw_episode(ax, col, title):
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if ep is not None:
            ax.plot(ep[x], ep[col], alpha=0.35, label="episode")
            ax.plot(ep[x], ep[col].rolling(args.window, min_periods=1).mean(),
                    label=f"moving avg ({args.window})")
            ax.legend()

    draw_episode(axes[0, 0], "total_reward", "Episode reward")
    draw_episode(axes[0, 1], "final_distance", "Episode final distance")

    ax = axes[0, 2]
    ax.set_title("Success rate")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    if ep is not None:
        ax.plot(ep[x], ep["success_rate_100"], label="train (100 ep)")
    if va is not None:
        ax.plot(va[x], va["success_rate"], marker="o", label="validation")
    ax.legend()

    draw(axes[1, 0], st, ["mean_reward_100"], "Mean reward (100 ep)")
    draw(axes[1, 1], st, ["actor_loss", "critic_loss"], "Actor / critic loss")
    draw(axes[1, 2], st, ["alpha"], "Entropy coefficient alpha")
    draw(axes[2, 0], va, ["mean_reward"], "Validation mean reward")
    draw(axes[2, 1], va, ["mean_distance"], "Validation mean distance")
    draw(axes[2, 2], st, ["steps_per_second"], "Steps per second")

    for ax in axes[2]:
        ax.set_xlabel("global step")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    if critics is not None or actions is not None:
        extra, diagnostic_axes = plt.subplots(3, 4, figsize=(20, 11), sharex=True)
        extra.suptitle(f"Critics and executed actions: {run_dir.name}")
        draw(diagnostic_axes[0, 0], critics, ["q1_mean", "q2_mean", "q_target_mean"], "Q values / targets")
        draw(diagnostic_axes[0, 1], critics, ["td_mse_terminal", "td_mse_nonterminal"], "TD squared errors (sum of 2 critics)")
        draw(diagnostic_axes[0, 2], critics, ["q_gap_mean", "td_abs_mean"], "Critic disagreement / absolute TD error")
        draw(diagnostic_axes[0, 3], critics, ["entropy_mean", "entropy_error"], "Policy entropy / gap to target")
        for i in range(4):
            draw(diagnostic_axes[1, i], actions, [f"action_{i}", f"effective_action_{i}", f"target_at_limit_{i}"],
                 f"Channel {i}: requested / effective / fraction at limit")
        draw(diagnostic_axes[2, 0], actions, [f"tracking_error_{i}" for i in range(4)], "Tracking error (arm: rad; fingers: m)")
        draw(diagnostic_axes[2, 1], actions, [f"action_saturated_{i}" for i in range(4)], "Fraction of actions with amplitude >= 0.95")
        draw(diagnostic_axes[2, 2], actions, [f"action_clipped_{i}" for i in range(4)], "Fraction of actions clipped by target limits")
        draw(diagnostic_axes[2, 3], critics, [f"policy_std_{i}" for i in range(4)], "Gaussian std before tanh")
        for ax in diagnostic_axes[2]:
            ax.set_xlabel("global step")
        extra.tight_layout()
        if args.save:
            destination = Path(args.save)
            diagnostic_path = destination.with_name(destination.stem + "_diagnostics" + destination.suffix)
            extra.savefig(diagnostic_path, dpi=150)
            print(f"Saved {diagnostic_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
        if critics is not None or actions is not None:
            plt.close(extra)


def plot_bc(run_dir, args):
    metrics = load(run_dir, 'metrics')
    if metrics is None:
        raise SystemExit(f'No BC metrics in {run_dir}')
    validation = load(run_dir, 'validation')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Behavior cloning: {run_dir.name}')
    for ax, metric in zip(axes[0], ('mse', 'mae')):
        for split in ('train', 'validation'):
            ax.plot(metrics['epoch'], metrics[f'{split}_{metric}'], label=split)
        ax.set_title(metric.upper())
        ax.legend()
    for channel in range(4):
        axes[1, 0].plot(metrics['epoch'], metrics[f'validation_mse_{channel}'], label=f'action {channel}')
    axes[1, 0].set_title('Validation MSE by action channel')
    axes[1, 0].legend()
    axes[1, 1].set_title('MuJoCo validation success rate')
    axes[1, 1].set_ylim(-.05, 1.05)
    if validation is not None:
        axes[1, 1].plot(validation['epoch'], validation['success_rate'], marker='o')
    else:
        axes[1, 1].text(.5, .5, 'no rollout data', ha='center', transform=axes[1, 1].transAxes)
    for ax in axes.flat:
        ax.set_xlabel('epoch')
        ax.grid(alpha=.3)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f'Saved {args.save}')
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
