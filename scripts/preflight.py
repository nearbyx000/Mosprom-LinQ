"""Readiness check: no camera opening, neural inference, or TPU job publication."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

from safesight.config import load_config
from safesight.safety import SafetyPolicy


def inspect(mode='preview',config_path='config/cameras.json',data_dir=None,network=False):
    checks = []
    def check(name,operation):
        try:
            detail = operation()
            checks.append({'check':name,'ok':True,'detail':detail})
        except Exception as exc:
            # Credentials / URLs / worker paths never appear in diagnostics.
            detail = str(exc) if isinstance(exc,(ValueError,FileNotFoundError)) else type(exc).__name__
            checks.append({'check':name,'ok':False,'detail':detail})
    check('python',lambda:sys.version.split()[0] if sys.version_info>=(3,11) else (_ for _ in ()).throw(ValueError('Python 3.11+ required; 3.12 recommended')))
    for package in ('fastapi','uvicorn','opencv-python','numpy','websockets'):
        check('dependency:'+package,lambda name=package:importlib.metadata.version(name))
    check('camera_config',lambda:len(load_config(config_path)['cameras']))
    check('safety_policy',lambda:SafetyPolicy.from_env().thresholds)
    def auth_settings():
        if bool(os.getenv('SAFESIGHT_AUTH_USER')) != bool(os.getenv('SAFESIGHT_AUTH_PASSWORD')):
            raise ValueError('Configure both operator auth values or neither')
        return 'enabled' if os.getenv('SAFESIGHT_AUTH_USER') else 'disabled; intended for local access'
    check('operator_auth',auth_settings)
    target = Path(data_dir or os.getenv('SAFESIGHT_DATA') or ('data/demo' if mode=='demo' else 'data'))
    def writable():
        target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=target) as stream:
            stream.write(b'SafeSight preflight')
            stream.flush()
        return 'writable; existing event data unchanged'
    check('data_directory',writable)
    if mode=='linq':
        check('dependency:pika',lambda:importlib.metadata.version('pika'))
        jpeg_mode = bool(os.getenv('LINQ_JPEG_ROUTE'))
        def linq_settings():
            from safesight.jpeg_gateway import JPEGSettings
            from safesight.rabbitmq import Settings
            settings = JPEGSettings.from_env() if jpeg_mode else Settings.from_env()
            if os.getenv('LINQ_SV_ROUTE'):
                if not jpeg_mode:
                    raise ValueError('Dual mode requires LINQ_JPEG_ROUTE as well as LINQ_SV_ROUTE')
                if os.getenv('LINQ_SV_ROUTE') == settings.route:
                    raise ValueError('Primary and SafetyVision gateway routes must be distinct')
                replace(settings,route=os.environ['LINQ_SV_ROUTE']).validate()
            return 'JPEG gateway' if jpeg_mode else 'direct tensor protocol'
        check('linq_settings',linq_settings)
        if not jpeg_mode:
            def codec():
                if importlib.util.find_spec('tpu_pycloud_format') is None:
                    raise ValueError('Install organizer-provided tpu_pycloud_format; no substitute codec is used')
                return 'installed'
            check('organizer_codec',codec)
        if network and all(c['ok'] for c in checks if c['check'] in {'linq_settings','dependency:pika'}):
            def connect_only():
                import pika
                connection = pika.BlockingConnection(pika.ConnectionParameters(
                    host=os.getenv('RABBITMQ_HOST','10.0.114.30'),port=int(os.getenv('RABBITMQ_PORT','5672')),
                    virtual_host=os.getenv('RABBITMQ_VHOST','/'),
                    credentials=pika.PlainCredentials(os.getenv('RABBITMQ_USERNAME','guest'),os.getenv('RABBITMQ_PASSWORD','guest')),
                    socket_timeout=3,stack_timeout=5,connection_attempts=1,blocked_connection_timeout=3))
                connection.close()
                return 'AMQP handshake succeeded; zero inference tasks published'
            check('amqp_connection',connect_only)
    if mode=='local':
        check('local_model',lambda:'present' if Path(os.getenv('SAFESIGHT_MODEL','')).is_file() else (_ for _ in ()).throw(ValueError('Set SAFESIGHT_MODEL to an existing local model')))
        check('dependency:ultralytics',lambda:importlib.metadata.version('ultralytics'))
    return {'ok':all(c['ok'] for c in checks),'mode':mode,'platform':platform.system(),
            'checks':checks,'camera_opened':False,'inference_tasks_published':0,
            'note':'Configuration readiness only; real camera and TPU accuracy require separate checks'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['preview','demo','linq','local'],default='demo')
    parser.add_argument('--config',default='config/cameras.json')
    parser.add_argument('--data-dir')
    parser.add_argument('--env-file')
    parser.add_argument('--network',action='store_true',help='AMQP connect/close only; no TPU job')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.env_file:
        if not Path(args.env_file).is_file():
            parser.error('Environment file does not exist')
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    report = inspect(args.mode,args.config,args.data_dir,args.network)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2,ensure_ascii=False)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())
