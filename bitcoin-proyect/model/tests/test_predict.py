# tests/test_predict.py
# cubre predict.py — el módulo más crítico y el que antes no tenía ningún test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pipeline(n_test: int = 20):
    """
    Construye un pipeline sintético que imita _load_pipeline()
    sin tocar disco ni red. Permite testear la lógica de predict.py
    de forma aislada.
    """
    from sklearn.preprocessing import MinMaxScaler

    prices   = np.linspace(10_000, 70_000, 500).reshape(-1, 1)
    scaler   = MinMaxScaler().fit(prices)
    scaled   = scaler.transform(prices)
    y_test   = scaled[-n_test:, 0]
    X_test   = np.stack([scaled[i : i + 60] for i in range(len(scaled) - 60 - n_test, len(scaled) - 60)])

    mock_model = MagicMock()
    mock_model.predict.return_value = y_test.reshape(-1, 1)

    import pandas as pd
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    df    = MagicMock()
    df.index = dates
    df["Close"].values = prices.flatten()

    return {
        "df":     df,
        "prices": prices,
        "scaled": scaled,
        "scaler": scaler,
        "model":  mock_model,
        "X_test": X_test,
        "y_test": y_test,
    }


# ── PT-15 — get_predictions devuelve listas de igual longitud ────────────────
# real y predicted deben tener el mismo número de elementos

def test_get_predictions_misma_longitud():
    import predict
    with patch.object(predict, "_load_pipeline", return_value=_make_pipeline()):
        predict._cache = None
        real, pred = predict.get_predictions()
    assert len(real) == len(pred)


# ── PT-16 — get_predictions devuelve precios en escala USD ───────────────────
# los valores deben estar en el rango de precios reales de Bitcoin (> 1000)

def test_get_predictions_escala_usd():
    import predict
    with patch.object(predict, "_load_pipeline", return_value=_make_pipeline()):
        predict._cache = None
        real, pred = predict.get_predictions()
    assert min(real) > 1_000
    assert min(pred) > 1_000


# ── PT-17 — get_metrics devuelve un float no negativo ────────────────────────

def test_get_metrics_valor_valido():
    import predict
    with patch.object(predict, "_load_pipeline", return_value=_make_pipeline()):
        predict._cache = None
        mae = predict.get_metrics()
    assert isinstance(mae, float)
    assert mae >= 0.0


# ── PT-18 — forecast_next_days devuelve la cantidad de días solicitada ────────

def test_forecast_longitud():
    import predict
    pipeline = _make_pipeline()
    with patch.object(predict, "_load_pipeline", return_value=pipeline):
        predict._cache = None
        fechas, precios = predict.forecast_next_days(days=15)
    assert len(fechas)  == 15
    assert len(precios) == 15


# ── PT-19 — forecast devuelve fechas en formato YYYY-MM-DD ───────────────────

def test_forecast_formato_fechas():
    import re
    import predict
    pipeline = _make_pipeline()
    with patch.object(predict, "_load_pipeline", return_value=pipeline):
        predict._cache = None
        fechas, _ = predict.forecast_next_days(days=5)
    patron = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    assert all(patron.match(f) for f in fechas)


# ── PT-20 — run_api_mode emite JSON válido para cada flag ─────────────────────

@pytest.mark.parametrize("flag,expected_keys", [
    ("--metrics",  ["mae"]),
    ("--predict",  ["real", "predicted"]),
    ("--history",  ["dates", "prices"]),
    ("--forecast", ["dates", "prices"]),
])
def test_run_api_mode_json_valido(flag, expected_keys, capsys):
    import predict
    pipeline = _make_pipeline()

    loss_data = json.dumps({"train": [0.1, 0.05], "val": [0.12, 0.06]})

    with (
        patch.object(predict, "_load_pipeline", return_value=pipeline),
        patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=lambda s, *a: MagicMock(read=lambda: loss_data),
                __exit__=MagicMock(return_value=False),
            )
        )),
    ):
        predict._cache = None
        predict.run_api_mode(flag)

    captured = capsys.readouterr()
    data     = json.loads(captured.out)
    for key in expected_keys:
        assert key in data


# ── PT-21 — flag inválido termina con código de salida 1 ──────────────────────

def test_run_api_mode_flag_invalido():
    import predict
    with pytest.raises(SystemExit) as exc_info:
        predict.run_api_mode("--no-existe")
    assert exc_info.value.code == 1