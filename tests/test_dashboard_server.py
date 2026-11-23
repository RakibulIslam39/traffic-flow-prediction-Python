"""Starts the real Streamlit web server and requests what a browser asks for.

``tests/test_dashboard.py`` runs the dashboard script without a web server, so it cannot see
server-side failures. One such failure happened on Intel Macs: an old Streamlit combined with
Starlette 1.5+ answered ``GET /favicon.png`` with HTTP 500 and logged "Expected ASGI message
'http.response.start', but got 'http.response.body'". This test would have caught it.
"""
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("streamlit")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "dashboard", "app.py")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port, path, encoding="identity"):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 headers={"Accept-Encoding": encoding})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as err:
        return err.code, b""


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = _free_port()
    log_path = tmp_path_factory.mktemp("streamlit") / "server.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", APP, "--server.headless", "true",
             "--server.address", "127.0.0.1", "--server.port", str(port),
             "--browser.gatherUsageStats", "false"],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    try:
        deadline = time.time() + 90
        while True:
            if proc.poll() is not None:
                pytest.fail("Streamlit exited early:\n" + log_path.read_text())
            try:
                if _get(port, "/_stcore/health")[0] == 200:
                    break
            except OSError:
                pass
            if time.time() > deadline:
                pytest.fail("Streamlit did not start in time:\n" + log_path.read_text())
            time.sleep(0.5)
        yield port, log_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_pages_and_assets_are_served_without_errors(server):
    port, log_path = server
    status, html = _get(port, "/")
    assert status == 200
    bundle = re.search(rb'static/js/index\.[\w-]+\.js', html)
    assert bundle, "main JavaScript bundle not referenced in the index page"

    for encoding in ("gzip, deflate, br", "identity"):
        for path in ("/favicon.png", "/manifest.json", "/" + bundle.group(0).decode()):
            status, body = _get(port, path, encoding)
            assert status == 200, "%s [%s] -> HTTP %d" % (path, encoding, status)
            assert body

    time.sleep(0.5)
    log = log_path.read_text()
    assert "Expected ASGI message" not in log
    assert "Traceback" not in log
