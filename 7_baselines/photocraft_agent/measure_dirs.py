"""实测 PhotoCraft filter.cameraRaw 关键参数的方向与强度（写 prompt 参数说明用，不凭直觉）。

色块：中灰 128、纯红、纯绿、纯蓝、肤色，各 32×32，横向拼成一张测试图。
"""
import colorsys
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PATCHES = {'gray': (128, 128, 128), 'red': (200, 40, 40), 'green': (50, 160, 60), 'blue': (40, 70, 200),
           'skin': (215, 160, 130), 'dark': (40, 40, 40), 'light': (225, 225, 225)}
CASES = [
    ('exposure', 1.0), ('exposure', -1.0), ('temperature', 30), ('temperature', -30), ('tint', 30), ('tint', -30),
    ('contrast', 50), ('highlights', -60), ('shadows', 60), ('whites', 50), ('blacks', -50),
    ('vibrance', 50), ('saturation', 50), ('saturation', -50), ('saturation', -100),
    ('hslHue', [40, 0, 0, 0, 0, 0, 0, 0]), ('hslHue', [0, 0, 0, 40, 0, 0, 0, 0]), ('hslHue', [0, 0, 0, 0, 0, 40, 0, 0]),
    ('hslSat', [-60, 0, 0, 0, 0, 0, 0, 0]), ('hslLum', [0, 0, 0, 0, 0, -60, 0, 0]),
    ('gradeShadows', {'hue': 220, 'sat': 60, 'lum': 0}),
]


def main():
    tmp = Path(tempfile.mkdtemp())
    img = np.concatenate([np.full((32, 32, 3), c, np.uint8) for c in PATCHES.values()], 1)
    Image.fromarray(img).save(tmp / 'in.png')
    for k, v in CASES:
        subprocess.run(['photocraft-cli', 'run', str(tmp / 'in.png'), '--cmd', 'filter.cameraRaw',
                        '--params', json.dumps({k: v}), '--out', str(tmp / 'out.png')], check=True, capture_output=True)
        out = np.asarray(Image.open(tmp / 'out.png').convert('RGB'))
        cells = []
        for i, name in enumerate(PATCHES):
            o = out[8:24, i * 32 + 8:i * 32 + 24].reshape(-1, 3).mean(0)
            h0 = colorsys.rgb_to_hsv(*(np.array(PATCHES[name]) / 255))[0] * 360
            h1 = colorsys.rgb_to_hsv(*(o / 255))[0] * 360
            cells.append(f'{name}{tuple(int(x) for x in PATCHES[name])}→{tuple(int(round(x)) for x in o)}'
                         + (f' hue {h0:.0f}→{h1:.0f}' if name in ('red', 'green', 'blue') and k == 'hslHue' else ''))
        print(f'{k}={json.dumps(v)}:\n   ' + '\n   '.join(cells))


if __name__ == '__main__':
    main()
