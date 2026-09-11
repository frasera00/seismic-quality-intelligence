from sklearn.ensemble import RandomForestClassifier


def build_model(config: dict) -> RandomForestClassifier:
    """Build a baseline model using an experiment configuration."""
    model_config = config["model"]

    if model_config["name"] != "random_forest":
        raise ValueError(f"Unsupported model: {model_config['name']}")

    return RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        random_state=model_config["random_state"],
        n_jobs=model_config["n_jobs"],
    )
