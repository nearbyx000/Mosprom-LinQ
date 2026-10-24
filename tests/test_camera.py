"""Preserve the camera backend fixes introduced concurrently on WEB."""
from unittest.mock import patch

import cv2
import pytest

from safesight import pipeline


@pytest.mark.parametrize("platform,backend_name", [
    ("win32", "CAP_DSHOW"),
    ("darwin", "CAP_AVFOUNDATION"),
    ("linux", None),
])
def test_native_camera_backend(platform, backend_name):
    with patch.object(pipeline.sys, "platform", platform), patch.object(pipeline.cv2, "VideoCapture") as capture:
        pipeline.open_camera("0", 30)
    expected = (0, getattr(cv2, backend_name)) if backend_name else (0,)
    capture.assert_called_once_with(*expected)


def test_windows_network_camera_uses_network_backend_with_timeouts():
    source = "rtsp://192.168.1.105/live"
    with patch.object(pipeline.sys, "platform", "win32"), patch.object(pipeline.cv2, "VideoCapture") as capture:
        pipeline.open_camera(source, 30)
    capture.assert_called_once_with(source, cv2.CAP_FFMPEG,
                                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
