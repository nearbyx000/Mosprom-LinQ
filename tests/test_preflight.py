import json

from scripts.preflight import inspect


def test_demo_preflight_does_not_open_camera_or_publish_tasks(tmp_path,monkeypatch):
    monkeypatch.delenv('SAFESIGHT_AUTH_USER',raising=False)
    monkeypatch.delenv('SAFESIGHT_AUTH_PASSWORD',raising=False)
    report = inspect('demo',data_dir=tmp_path/'data')
    assert report['ok'] is True
    assert report['camera_opened'] is False and report['inference_tasks_published'] == 0


def test_bad_auth_configuration_is_reported_without_secrets(tmp_path,monkeypatch):
    monkeypatch.setenv('SAFESIGHT_AUTH_PASSWORD','not-a-real-password')
    monkeypatch.delenv('SAFESIGHT_AUTH_USER',raising=False)
    report = inspect('demo',data_dir=tmp_path/'data')
    assert report['ok'] is False
    assert 'not-a-real-password' not in json.dumps(report)


def test_duplicate_routes_fail_before_network(tmp_path,monkeypatch):
    monkeypatch.setenv('LINQ_JPEG_ROUTE','safesight.jpeg.same')
    monkeypatch.setenv('LINQ_SV_ROUTE','safesight.jpeg.same')
    report = inspect('linq',data_dir=tmp_path/'data',network=True)
    assert report['ok'] is False
    assert not any(c['check']=='amqp_connection' for c in report['checks'])
