"""Generate a local LLM analysis of seismic-QC run artifacts with Ollama."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import ollama
import yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate an evidence-grounded Markdown report from saved "
            "seismic-QC artifacts using a local Ollama vision model."
        )
    )

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing training and calibration artifacts.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the experiment configuration.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path(
            "reports/figures/gather_prediction_overlay.png"
        ),
        help="Prediction-overlay figure to analyze.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/local_llm_analysis.md"),
        help="Markdown report written by this script.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2-vision",
        help="Installed Ollama vision-model name.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in: {path}")

    return content


def yaml_block(
    title: str,
    content: dict[str, Any],
) -> str:
    """Render one YAML artifact for the prompt."""
    rendered = yaml.safe_dump(
        content,
        sort_keys=False,
        allow_unicode=True,
    )

    return f"## {title}\n```yaml\n{rendered}```"


def build_prompt(
    config: dict[str, Any],
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    """Create an open-ended but evidence-bound analyst prompt."""
    artifacts = "\n\n".join(
        [
            yaml_block("Experiment configuration", config),
            yaml_block("Training metrics", metrics),
            yaml_block("Calibration report", calibration),
            yaml_block("Operating policy", policy),
        ]
    )

    instructions = """
You are reviewing one experiment from a seismic trace-quality anomaly-detection
project. You have been given configuration, metrics, calibration artifacts,
operating policy, and a prediction-overlay figure.

Independently identify the most important patterns. You have not been told
which pattern to look for.

Write a concise, evidence-grounded Markdown report using exactly these sections:

# Experiment Analysis

## Summary

## Observations

## Interpretations

## Limitations

## Recommended Experiments

Rules:
- Treat supplied numeric artifacts as authoritative.
- Do not invent metric values, model features, anomaly types, data-processing
  steps, or causal explanations.
- Clearly distinguish direct observations from interpretations.
- For each interpretation, state why it is plausible and state uncertainty.
- Do not claim that visual appearance alone proves a causal mechanism.
- Do not claim generalization to real field seismic data; this experiment uses
  synthetic data.
- Each recommended experiment must include a measurable success criterion.
- Discuss both what worked and what failed or remains uncertain.
""".strip()

    return f"{instructions}\n\n{artifacts}"


def main() -> None:
    """Read artifacts, call local Ollama, and write a Markdown report."""
    args = parse_args()

    if not args.figure_path.exists():
        raise FileNotFoundError(
            f"Missing figure to analyze: {args.figure_path}"
        )

    config = load_yaml(args.config_path)
    metrics = load_yaml(
        args.reports_dir / "gather_qc_metrics.yaml"
    )
    calibration = load_yaml(
        args.reports_dir / "calibrated_thresholds.yaml"
    )
    policy = load_yaml(
        args.reports_dir / "operating_policy.yaml"
    )

    prompt = build_prompt(
        config=config,
        metrics=metrics,
        calibration=calibration,
        policy=policy,
    )

    response = ollama.chat(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(args.figure_path)],
            }
        ],
        options={
            "temperature": 0.2,
        },
    )

    report = response["message"]["content"].strip()

    if not report:
        raise RuntimeError("The local LLM returned an empty report.")

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_path.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(f"Saved local LLM report: {args.output_path}")


if __name__ == "__main__":
    main()