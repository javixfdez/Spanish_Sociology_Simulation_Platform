import hashlib
import http.server
import json
import threading
import zipfile

import pytest

from mavericks import assets

REL = "outputs/models/prior_mlp.json"


def make_zip(path, content=b'{"a": 1}'):
    manifest = {"files": [{"path": REL, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(REL, content)
        z.writestr("manifest.json", json.dumps(manifest))
    return path


@pytest.fixture
def server(tmp_path):
    zip_path = make_zip(tmp_path / "datos.zip")
    (tmp_path / "pagina.html").write_text("<html>no es un zip</html>", encoding="utf-8")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(tmp_path), **k)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", zip_path
    httpd.shutdown()


def test_download_from_plain_url_and_verify_manifest(server, tmp_path, monkeypatch):
    base, _ = server
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAVERICKS_ASSETS_URL", raising=False)
    out = assets.fetch_assets(url=f"{base}/datos.zip", dest=str(tmp_path / "dest"))
    assert out["verified"] and (tmp_path / "dest" / REL).read_bytes() == b'{"a": 1}'


def test_url_that_returns_a_web_page_gives_a_clear_error(server, tmp_path):
    base, _ = server
    with pytest.raises(RuntimeError, match="página web|zip"):
        assets.download_url(f"{base}/pagina.html", tmp_path / "x.zip")


def test_drive_file_id_from_the_usual_links():
    fid = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    for url in (f"https://drive.google.com/file/d/{fid}/view?usp=sharing", f"https://drive.google.com/open?id={fid}", f"https://drive.google.com/uc?export=download&id={fid}",
                f"https://drive.usercontent.google.com/download?id={fid}&export=download"):
        assert assets.drive_file_id(url) == fid
    assert assets.drive_file_id("https://ejemplo.org/datos.zip") is None


class FakeResponse:
    def __init__(self, body=b"", content_type="application/zip", text=""):
        self.body, self.headers, self.text = body, {"Content-Type": content_type, "Content-Length": str(len(body))}, text

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.body


def test_google_drive_virus_warning_page_is_followed(tmp_path, monkeypatch):
    zip_path = make_zip(tmp_path / "datos.zip")
    page = ('<html><form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'
            '<input type="hidden" name="id" value="FILEID"><input type="hidden" name="export" value="download"><input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="u-1"></form></html>')
    calls = []

    def fake_get(self, url, params=None, **kwargs):
        calls.append((url, params))
        if len(calls) == 1:
            return FakeResponse(content_type="text/html; charset=utf-8", text=page)
        return FakeResponse(body=zip_path.read_bytes())

    monkeypatch.setattr("requests.Session.get", fake_get)
    target = assets.download_url("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz012345/view?usp=sharing", tmp_path / "bajado.zip")
    assert zipfile.is_zipfile(target) and len(calls) == 2 and calls[1][1]["uuid"] == "u-1" and calls[1][1]["confirm"] == "t"


def test_url_comes_from_env_file_and_failures_list_the_three_ways(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAVERICKS_ASSETS_URL", raising=False)
    (tmp_path / ".env").write_text('OTRA=1\nMAVERICKS_ASSETS_URL="http://127.0.0.1:9/datos.zip"\n', encoding="utf-8")
    assert assets.env_value("MAVERICKS_ASSETS_URL") == "http://127.0.0.1:9/datos.zip"

    class NoHub:
        @staticmethod
        def hf_hub_download(**kwargs):
            raise PermissionError("401 sin acceso")

    monkeypatch.setattr(assets, "_hub", lambda: NoHub)
    monkeypatch.setattr(assets, "DEFAULT_URL", "http://127.0.0.1:9/copia.zip")  # sin red: la copia pública de Drive también falla
    with pytest.raises(SystemExit) as exc:
        assets.get_zip(tmp_dir=tmp_path)
    message = str(exc.value)
    assert "enlace" in message and "Hugging Face" in message and "zip local" in message and "Hay tres vías" in message


def test_without_huggingface_access_the_public_drive_copy_is_the_last_resort(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MAVERICKS_ASSETS_URL", raising=False)

    class NoHub:
        @staticmethod
        def hf_hub_download(**kwargs):
            raise PermissionError("401 sin acceso")

    calls = []

    def fake_download(url, target):
        calls.append(url)
        return make_zip(target)

    monkeypatch.setattr(assets, "_hub", lambda: NoHub)
    monkeypatch.setattr(assets, "download_url", fake_download)
    path = assets.get_zip(tmp_dir=tmp_path)
    assert calls == [assets.DEFAULT_URL] and zipfile.is_zipfile(path)
