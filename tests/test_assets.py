import hashlib
import json
import zipfile

import pytest

from mavericks import assets


def make_zip(path, files, tamper=False):
    manifest = {"files": [{"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in files.items()]}
    if tamper:
        manifest["files"][0]["sha256"] = "0" * 64
    with zipfile.ZipFile(path, "w") as z:
        for p, b in files.items():
            z.writestr(p, b)
        z.writestr("manifest.json", json.dumps(manifest))
    return path


def test_only_closed_list_and_never_raw_or_original_formats():
    assets.check_allowed("data/processed/cis_respondents.parquet")
    for bad in ("maverick_data/raw/x.parquet", "data/processed/original.sav", "data/processed/ZA7848.dta", "data/processed/otro.parquet", "../fuera.json", "/abs/x.json"):
        with pytest.raises(AssertionError):
            assets.check_allowed(bad)
    assert not any(p.startswith("maverick_data") for p, *_ in assets.ASSETS)
    assert all(not p.endswith(tuple(assets.FORBIDDEN_SUFFIXES)) for p, *_ in assets.ASSETS)


def test_fetch_assets_from_local_zip_verifies_sha(tmp_path):
    rel = "outputs/models/prior_mlp.json"
    good = make_zip(tmp_path / "ok.zip", {rel: b'{"a": 1}'})
    out = assets.fetch_assets(str(good), dest=str(tmp_path / "dest"))
    assert out["verified"] and (tmp_path / "dest" / rel).read_bytes() == b'{"a": 1}'
    bad = make_zip(tmp_path / "bad.zip", {rel: b'{"a": 1}'}, tamper=True)
    with pytest.raises(SystemExit, match="sha256"):
        assets.fetch_assets(str(bad), dest=str(tmp_path / "dest2"))


def test_fetch_assets_rejects_paths_outside_the_closed_list(tmp_path):
    evil = make_zip(tmp_path / "evil.zip", {"../escape.txt": b"x"})
    with pytest.raises(AssertionError):
        assets.fetch_assets(str(evil), dest=str(tmp_path / "dest"))
    assert not (tmp_path / "escape.txt").exists()


def test_filter_national_removes_reserved(tmp_path):
    src = tmp_path / "n.json"
    src.write_text(json.dumps({"CIS1_A": {"n": 1}, "CIS9_B": {"n": 2}}), encoding="utf-8")
    kept, dropped = assets.filter_national(src, tmp_path / "o.json", {"CIS9_B"})
    assert (kept, dropped) == (1, 1) and list(json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))) == ["CIS1_A"]


def test_adapter_sha_mismatch_is_rejected(tmp_path, monkeypatch):
    class Hub:
        @staticmethod
        def snapshot_download(repo_id, repo_type, local_dir, allow_patterns):
            (tmp_path / "ad").mkdir(exist_ok=True)
            (tmp_path / "ad" / "adapter_model.safetensors").write_bytes(b"otro")

    monkeypatch.setattr(assets, "_hub", lambda: Hub)
    with pytest.raises(SystemExit, match="no es el esperado"):
        assets.fetch_adapter(dest=str(tmp_path / "ad"))
