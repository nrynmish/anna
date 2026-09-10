from pathlib import Path
import tomllib


def test_project_metadata_is_configured():
    root = Path(__file__).resolve().parents[1]
    pyproject_path = root / "pyproject.toml"

    assert pyproject_path.exists()

    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)

    assert data["project"]["name"] == "anna"
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "tool.pytest.ini_options" in pyproject_path.read_text(encoding="utf-8")
