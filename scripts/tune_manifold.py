"""Run persistent single-study Optuna TPESampler tuning over SubStateManifold hyperparameters."""

from yaw.daemon.manifold_optuna import main

if __name__ == "__main__":
    raise SystemExit(main())
