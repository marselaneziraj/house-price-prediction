from pathlib import Path

def test_required_paths_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "README.md",
        root / "PROJECT_PROMPT.md",
        root / "requirements.txt",
        root / "src" / "config.py",
        root / "src" / "data" / "load_data.py",
        root / "src" / "data" / "make_dataset.py",
        root / "src" / "features" / "feature_engineering.py",
        root / "src" / "features" / "preprocess.py",
        root / "src" / "models" / "train.py",
        root / "src" / "models" / "evaluate.py",
        root / "src" / "models" / "tune_random_forest.py",
        root / "src" / "pipelines" / "training_pipeline.py",
    ]
    for path in required:
        assert path.exists(), f"Missing required file: {path}"
