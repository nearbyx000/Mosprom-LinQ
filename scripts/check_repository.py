"""Check the reviewable Git tree for missing docs links and accidental artifacts."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote,urlsplit


def check(root):
    root = Path(root)
    result = subprocess.run(['git','ls-files','-z'],cwd=root,capture_output=True,check=True)
    files = [Path(p) for p in result.stdout.decode('utf-8').split('\0') if p]
    errors = []
    forbidden = {'.onnx','.pt','.tpu','.zip','.sqlite','.db','.ovpn','.pem','.key'}
    patterns = [re.compile(r'(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----$'),
                re.compile(r'ghp_[A-Za-z0-9]{36}'),re.compile(r'AKIA[A-Z0-9]{16}')]
    for relative in files:
        path = root/relative
        if not path.exists():
            errors.append(f'Missing indexed file: {relative}')
            continue
        if relative.suffix.lower() in forbidden or relative.name=='.env' or relative.parts[0] in {'data','.venv'}:
            errors.append(f'Local artifact indexed: {relative}')
        if relative.suffix.lower() in {'.py','.js','.json','.md','.yml','.yaml','.txt','.example'} and path.stat().st_size<2_000_000:
            content = path.read_text(encoding='utf-8-sig')
            if any(pattern.search(content) for pattern in patterns):
                errors.append(f'Potential credential in {relative}; inspect locally, value suppressed')
        if relative.suffix.lower()=='.md' and 'legacy' not in relative.parts:
            content = path.read_text(encoding='utf-8-sig')
            for link in re.findall(r'\]\(([^)]+)\)',content):
                target = link.strip('<>').split('#',1)[0]
                if not target or urlsplit(target).scheme or target.startswith('/'):
                    continue
                if not (path.parent/unquote(target)).exists():
                    errors.append(f'Broken document link in {relative}: {target}')
    return errors


def main():
    root = Path(__file__).resolve().parents[1]
    errors = check(root)
    for error in errors:
        print(error)
    if not errors:
        print('Repository checks passed: indexed source, document links, no model/SDK/data artifacts or obvious credentials.')
    return bool(errors)


if __name__=='__main__':
    raise SystemExit(main())
