"""
Guardrails for keeping the notebook, extracted training script, and refresh
workflow aligned on critical training settings.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from pipelines.train_model import FINAL_FEATURE_COLS, run as train_model_run


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ufc_pipeline.ipynb"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "refresh_fighters.yml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


def _read_notebook_source() -> str:
    with open(NOTEBOOK_PATH) as f:
        notebook = json.load(f)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_refresh_workflow_uses_tuned_training_defaults():
    workflow_text = WORKFLOW_PATH.read_text()
    notebook_source = _read_notebook_source()

    assert 'default: "25"' in workflow_text
    assert 'echo "tune_trials=25"' in workflow_text
    assert inspect.signature(train_model_run).parameters["tune_trials"].default == 25
    assert "study.optimize(objective, n_trials=25)" in notebook_source
    assert "optuna>=3" in REQUIREMENTS_PATH.read_text()


def test_refresh_workflow_avoids_rebase_push_conflicts():
    workflow_text = WORKFLOW_PATH.read_text()

    assert "git rebase origin/main" not in workflow_text
    assert "peter-evans/create-pull-request@v7" in workflow_text
    assert "actions/cache@v4" in workflow_text


def test_notebook_still_matches_critical_training_flow():
    notebook_source = _read_notebook_source()

    assert "TimeSeriesSplit(n_splits=5)" in notebook_source
    assert 'IsotonicRegression(out_of_bounds="clip")' in notebook_source
    assert 'df["wins_before"] = df.groupby("fighter")["target"].transform(' in notebook_source
    assert 'lambda s: s.shift(1).rolling(3).mean()' in notebook_source
    assert 'for col in missing_cols:' in notebook_source
    assert 'df[f"{col}_missing"] = df[col].isna().astype(int)' in notebook_source

    for col in [
        "rating_diff",
        "RD_diff",
        "height_diff",
        "reach_diff",
        "age_diff",
        "is_debut",
        "opp_is_debut",
    ]:
        assert f'"{col}"' in notebook_source
