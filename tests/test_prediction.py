from pathlib import Path
import __main__
import importlib.util

import joblib


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"
MODEL_PATH = ROOT / "models" / "modelo_final.joblib"


def load_app_module():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_predictions_match_expected_labels():
    app = load_app_module()
    __main__.PdaysTransformer = app.PdaysTransformer
    __main__.MaritalImputer = app.MaritalImputer
    model = joblib.load(MODEL_PATH)

    for demo, expected in (
        (app.INSTANCIA_DEMO_1, "no"),
        (app.INSTANCIA_DEMO_2, "yes"),
    ):
        frame = app.construir_dataframe(demo["data"])
        _, probability = app.predecir(model, frame)
        assert ("yes" if probability >= 0.5 else "no") == expected
