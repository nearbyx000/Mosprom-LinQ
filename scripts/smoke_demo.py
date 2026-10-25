"""Real HTTP smoke of an isolated synthetic demo; no physical camera or TPU.

Starts its own child server with explicit mode/source/data directory, checks a
whole scenario cycle and immutable evidence, then stops only that child. Keeps
the new output directory and log for inspection. Runtime requirements suffice.
Waits for the observed scenes within a bounded budget instead of asserting a
fixed frame rate from shared CI runners.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def run(output, duration=35, timeout=None, source_fps=30):
    if not 30 <= duration <= 300:
        raise ValueError('Minimum observation duration must be 30..300 seconds')
    timeout = max(120,duration*2) if timeout is None else timeout
    if not duration <= timeout <= 600 or source_fps not in (15,30):
        raise ValueError('Completion timeout must be duration..600 seconds; source FPS must be 15 or 30')
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('SAFESIGHT_','CAMERA_','LINQ_','RABBITMQ_'))}
    environment['CAMERA_FPS'] = str(source_fps)
    def get(path):
        with urlopen(base+path,timeout=5) as response:
            return response.read(),{key.lower():value for key,value in response.headers.items()}
    def get_json(path):
        return json.loads(get(path)[0])
    process = None
    try:
        with (output/'server.log').open('wb') as log:
            process = subprocess.Popen([sys.executable,str(root/'run.py'),'--mode','demo','--source','synthetic',
                                        '--data-dir',str(output/'demo-data'),'--port',str(port)],
                                       cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            startup = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Demo server exited during startup; inspect server.log')
                try:
                    status = get_json('/api/status')
                    if status['cameras'][0]['preview_available']:
                        break
                except (URLError,TimeoutError):
                    pass
                if time.monotonic()-startup>20:
                    raise RuntimeError('Demo startup timed out')
                time.sleep(.2)
            if status['mode']!='demo' or status['cameras'][0]['tpu_response_received']:
                raise RuntimeError('Expected explicitly synthetic demo, not hardware inference')
            camera = status['cameras'][0]['id']
            with urlopen(base+f'/api/stream/{camera}',timeout=5) as stream:
                packet = stream.read(65536)
                if b'Content-Type: image/jpeg' not in packet or b'X-Frame-Sequence:' not in packet:
                    raise RuntimeError('MJPEG stream contract missing')
            started = time.monotonic()
            samples = []
            clear_seen = False
            required_types = {'restricted_zone','no_helmet','fall_detected'}
            while True:
                current = get_json('/api/status')['cameras'][0]
                if current['status']!='running' or current['event_status']!='ready':
                    raise RuntimeError('Capture or storage failed during smoke')
                samples.append({'preview_fps':current['metrics']['fps'].get('preview'),
                                'detections_fps':current['metrics']['fps'].get('detections'),
                                'preview_age_ms':current['preview_age_ms'],
                                'frame':current['metrics']['counts'].get('capture',0)})
                clear_seen |= (current.get('demo_scenario')=='clear' and current['warnings_fresh']
                               and not current['active_warnings'])
                elapsed = time.monotonic()-started
                if elapsed >= duration:
                    rows = get_json('/api/events?limit=500')
                    types = {row['violation_type'] for row in rows}
                    if required_types <= types and clear_seen:
                        break
                    if elapsed >= timeout:
                        diagnostic = {'ok':False,'elapsed_seconds':round(elapsed,2),
                                      'missing_event_types':sorted(required_types-types),
                                      'clear_scene_observed':clear_seen,'scenario':current.get('demo_scenario'),
                                      'counts':current['metrics']['counts'],'fps':current['metrics']['fps']}
                        (output/'failure.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
                        raise RuntimeError('Demo completion timed out: '+json.dumps(diagnostic))
                time.sleep(1)
            for row in rows:
                image,headers = get('/evidence/'+row['evidence_file'])
                if row['source']!='demo' or hashlib.sha256(image).hexdigest()!=row['evidence_sha256']:
                    raise RuntimeError('Event source or immutable evidence checksum mismatch')
                if not headers.get('content-type','').startswith('image/jpeg'):
                    raise RuntimeError('Evidence is not a JPEG')
            if len({row['event_key'] for row in rows})!=len(rows):
                raise RuntimeError('Duplicate incident keys')
            csv_bytes,_ = get('/api/events/export.csv?limit=500')
            exported = list(csv.reader(io.StringIO(csv_bytes.decode('utf-8-sig'))))
            # History remains live between the JSON and CSV requests. New
            # events are allowed; every event in the earlier snapshot must
            # still be exported with exactly its saved checksum and source.
            indexed = {row[0]:row for row in exported[1:]}
            if (not csv_bytes.startswith(b'\xef\xbb\xbf') or not {str(row['id']) for row in rows} <= indexed.keys()
                    or exported[0][11:13]!=['SHA256 JPEG','Проверка оператора']
                    or any(row[7]!='demo' for row in exported[1:])
                    or any(indexed[str(row['id'])][11]!=row['evidence_sha256'] for row in rows)):
                raise RuntimeError('CSV contract missing')
            summary = get_json('/api/summary')
            drain_deadline = time.monotonic()+3
            while summary['outbox_pending'] and not summary['outbox_invalid'] and time.monotonic()<drain_deadline:
                time.sleep(.05)
                summary = get_json('/api/summary')
            if summary['outbox_pending'] or summary['outbox_invalid']:
                raise RuntimeError('Evidence remains unrecorded')
            if samples[-1]['frame']<=samples[0]['frame']:
                raise RuntimeError('Video stopped advancing')
            result = {'ok':True,'mode':'SYNTHETIC DEMO / NOT NEURAL / NOT TPU',
                      'duration_seconds':round(time.monotonic()-started,2),
                      'minimum_observation_seconds':duration,'completion_timeout_seconds':timeout,
                      'configured_source_fps':source_fps,'clear_scene_observed':clear_seen,
                      'events':len(rows),'event_types':sorted(types),'evidence_checksums_verified':len(rows),
                      'outbox_pending':0,'outbox_invalid':0,'samples':samples,
                      'note':'Local integration smoke, not neural quality or industrial performance measurement'}
            (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            return result
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New isolated directory; never an existing event store')
    parser.add_argument('--duration',type=float,default=35,help='Minimum observation period, not an FPS requirement')
    parser.add_argument('--timeout',type=float,help='Bounded completion budget, default max(120, 2*duration) seconds')
    parser.add_argument('--source-fps',type=int,choices=[15,30],default=30,help='15 simulates slower demo progression')
    args = parser.parse_args()
    result = run(args.output,args.duration,args.timeout,args.source_fps)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':
    main()
