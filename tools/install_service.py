"""Install a manually started systemd user service for this checkout."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'ur5-pi05-inference.service'


def quote(value):
    # systemd expands percent specifiers even inside double quotes.
    value = str(value)
    if '\n' in value or '\r' in value:
        raise ValueError('Service paths must fit on a single line')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Render a unit for inspection without installing it')
    args = parser.parse_args()
    command = ' '.join(quote(value) for value in [
        '/bin/bash', ROOT / 'run.sh', ROOT / 'serve.py',
        '--checkpoint', ROOT / '.runtime/checkpoint-current',
        '--warmup-sample', ROOT / '.runtime/warmup.npz',
        '--attention', 'sdpa', '--compile-mode', 'default', '--compile-all', '--port', '8000'])
    unit = '\n'.join([
        '[Unit]', 'Description=UR5 complete pi0.5 inference on AGX Orin', '',
        '[Service]', 'Type=simple', f'WorkingDirectory={str(ROOT).replace("%", "%%")}',
        'Environment=PYTHONUNBUFFERED=1', 'Environment=PYTHONFAULTHANDLER=1',
        'CPUWeight=50', 'IOWeight=50',
        f'StandardOutput=append:{str(ROOT).replace("%", "%%")}/.runtime/logs/service.log',
        'StandardError=inherit', f'ExecStart={command}',
        'Restart=on-failure', 'RestartSec=5', 'LimitCORE=0', ''])
    destination = args.output or Path.home() / '.config/systemd/user' / SERVICE
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not args.output and destination.exists() and destination.read_text() != unit:
        raise FileExistsError(f'A different service already exists: {destination}')
    destination.write_text(unit)
    if not args.output:
        (ROOT / '.runtime/logs').mkdir(parents=True, exist_ok=True)
        subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    print(f'Wrote {destination}')


if __name__ == '__main__':
    main()
