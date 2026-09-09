#!/usr/bin/env python
"""Shim: `python convert_relief.py album.png --depth-map d.png ...` runs scripts/convert_relief.py."""
import os
import runpy
import sys

if __name__ == '__main__':
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'convert_relief.py')
    sys.argv[0] = target
    runpy.run_path(target, run_name='__main__')
