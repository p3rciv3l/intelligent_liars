from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "fetch_tinylora_preservation_snapshot.py"
    )
    spec = importlib.util.spec_from_file_location("preservation_snapshot_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dataset_viewer_request_retries_rate_limit_and_caches_success(
    tmp_path: Path, monkeypatch
):
    module = _module()
    limited = Mock()
    limited.status_code = 429
    limited.headers = {"Retry-After": "0"}
    limited.raise_for_status.side_effect = module.requests.HTTPError("rate limited")
    success = Mock()
    success.status_code = 200
    success.headers = {}
    success.raise_for_status.return_value = None
    success.json.return_value = {"rows": [], "num_rows_total": 1}
    get = Mock(side_effect=[limited, success])
    monkeypatch.setattr(module.requests, "get", get)
    cache_path = tmp_path / "page.json"

    result = module._get_json(
        "rows", {"dataset": "example/data"}, cache_path=cache_path
    )
    cached = module._get_json(
        "rows", {"dataset": "example/data"}, cache_path=cache_path
    )

    assert result == {"rows": [], "num_rows_total": 1}
    assert cached == result
    assert get.call_count == 2
