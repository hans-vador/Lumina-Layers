"""Re-cache the X2D 0.08 mm system process preset from the installed Bambu Studio.

    cd ~/Lumina-Layers && .venv/bin/python scripts/refresh_x2d_process_preset.py

Reads ~/Library/Application Support/BambuStudio/system/BBL/process/*.json, resolves
the inherits chain of '0.08mm High Quality @BBL X2D' and writes
assets/x2d_process_0.08mm_high_quality_system.json, which core.stack5.pipeline uses
to build a self-contained process preset (see apply_stack5_process_preset).
Run it after a Bambu Studio update if you want the newest system values.
"""
import json
import os
import sys

SYSTEM = os.path.expanduser('~/Library/Application Support/BambuStudio/system/BBL')
PRESET = '0.08mm High Quality @BBL X2D'
META = {'name', 'from', 'setting_id', 'instantiation', 'inherits', 'type', 'version', 'is_custom_defined',
        'print_settings_id', 'compatible_printers', 'compatible_printers_condition', 'compatible_prints',
        'compatible_prints_condition', 'filament_settings_id', 'printer_settings_id'}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets',
                   'x2d_process_0.08mm_high_quality_system.json')


def resolve(kind, name, chain):
    with open(os.path.join(SYSTEM, kind, name + '.json'), encoding='utf-8') as fh:
        d = json.load(fh)
    chain.append(name)
    parent = d.get('inherits')
    base = resolve(kind, parent, chain) if parent else {}
    base.update({k: v for k, v in d.items() if k != 'inherits'})
    return base


def main():
    if not os.path.isdir(SYSTEM):
        sys.exit(f"Bambu Studio system profiles not found at {SYSTEM}")
    chain = []
    proc = resolve('process', PRESET, chain)
    with open(SYSTEM + '.json', encoding='utf-8') as fh:
        version = json.load(fh).get('version')
    out = {'source': 'Bambu Studio system profile chain (installed vendor profiles)',
           'studio_profile_version': version, 'preset': PRESET, 'chain': chain,
           'compatible_printers': proc.get('compatible_printers'),
           'config': {k: v for k, v in sorted(proc.items()) if k not in META}}
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}: {len(out['config'])} keys, profile version {version}, chain {' -> '.join(chain)}")


if __name__ == '__main__':
    main()
