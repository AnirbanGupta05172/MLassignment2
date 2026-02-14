"""
Auto-update README.md with the metrics table and observation template.

Run automatically after training, or manually:
    python -m model.make_readme
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"

TABLE_START = "<!-- METRICS_TABLE_START -->"
TABLE_END = "<!-- METRICS_TABLE_END -->"

def _to_markdown_table(metrics_df: pd.DataFrame) -> str:
    # Round to 4 decimals for readability
    df = metrics_df.copy()
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        df[col] = df[col].astype(float).round(4)
    return df.to_markdown(index=False)

def update_readme_with_metrics(metrics_df: pd.DataFrame) -> None:
    text = README_PATH.read_text(encoding="utf-8")

    if TABLE_START not in text or TABLE_END not in text:
        raise ValueError("README markers not found. Do not remove METRICS_TABLE markers.")

    md_table = _to_markdown_table(metrics_df)

    new_block = f"{TABLE_START}\n\n{md_table}\n\n{TABLE_END}"
    before = text.split(TABLE_START)[0]
    after = text.split(TABLE_END)[1]

    README_PATH.write_text(before + new_block + after, encoding="utf-8")

def main():
    from model.train_models import ART_DIR
    metrics_path = ART_DIR / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("metrics_summary.csv not found. Run training first.")
    metrics_df = pd.read_csv(metrics_path)
    update_readme_with_metrics(metrics_df)
    print("README.md updated with metrics table.")

if __name__ == "__main__":
    main()
