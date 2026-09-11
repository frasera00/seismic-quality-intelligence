import argparse

from scientific_ml_template.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print(f"Project: {config['project']['name']}")
    print(f"Model: {config['model']['name']}")
    print("Template initialized successfully.")


if __name__ == "__main__":
    main()
