import os

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "dashboard", "app.py")


def test_dashboard_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Traffic flow forecast"
    assert len(at.metric) == 5                      # now + four horizons
    assert len(at.dataframe) == 1                   # accuracy table


def test_dashboard_reacts_to_horizon_and_time():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.selectbox[1].set_value(60).run()
    assert not at.exception
    assert any("60 minutes ahead" in s.value for s in at.subheader)
    # a time with no full hour of readings before it on that day -> a warning, not a crash
    at.sidebar.slider[0].set_value(1.0).run()
    assert not at.exception
