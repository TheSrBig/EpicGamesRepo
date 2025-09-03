# setup.py — versión “todo a Drive”
# Guarda repos, modelos, entorno portable, binarios y temporales en:
#   /content/drive/MyDrive/BigFutureWUI
#
# UIs soportados: A1111, Forge, ReForge, Forge-Classic, ComfyUI, SwarmUI

from IPython.display import display, Image, clear_output
from IPython import get_ipython
from ipywidgets import widgets
from pathlib import Path
import subprocess
import argparse
import shlex
import json
import sys
import os
import re

# ---------------------------
# Utilidades y configuración
# ---------------------------

SyS = get_ipython().system
CD = os.chdir
iRON = os.environ

REPO = {
    'A1111': 'https://github.com/AUTOMATIC1111/stable-diffusion-webui A1111',
    'Forge': 'https://github.com/lllyasviel/stable-diffusion-webui-forge Forge',
    'ReForge': '-b main-old https://github.com/Panchovix/stable-diffusion-webui-reForge ReForge',
    'Forge-Classic': 'https://github.com/Haoming02/sd-webui-forge-classic Forge-Classic',
    'ComfyUI': 'https://github.com/comfyanonymous/ComfyUI',
    'SwarmUI': 'https://github.com/mcmonkeyprojects/SwarmUI'
}

WEBUI_LIST = ['A1111', 'Forge', 'ReForge', 'Forge-Classic', 'ComfyUI', 'SwarmUI']

def in_colab():
    try:
        import google.colab  # noqa
        return True
    except Exception:
        return False

def mount_drive_if_needed():
    if not in_colab():
        return False
    drive_root = Path('/content/drive')
    if not drive_root.exists() or not any(drive_root.iterdir()):
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
        except Exception as e:
            print(f'No fue posible montar Google Drive: {e}')
            return False
    return True

def getENV():
    # Solo usamos Colab para este flujo (todo a Drive).
    # Si no es Colab, abortamos.
    if in_colab():
        return 'Colab', '/content', '/content'
    return None, None, None

def getArgs():
    parser = argparse.ArgumentParser(description='WebUI Installer Script (Drive-first for Google Colab)')
    parser.add_argument('--webui', required=True, help='available webui: A1111, Forge, ReForge, Forge-Classic, ComfyUI, SwarmUI')
    parser.add_argument('--civitai_key', required=True, help='your CivitAI API key')
    parser.add_argument('--hf_read_token', default=None, help='your Huggingface READ Token (optional)')

    args, unknown = parser.parse_known_args()

    arg1 = args.webui.strip()
    arg2 = args.civitai_key.strip()
    arg3 = args.hf_read_token.strip() if args.hf_read_token else ''

    if not any(arg1.lower() == option.lower() for option in WEBUI_LIST):
        print(f'{ERROR}: invalid webui option: "{args.webui}"')
        print(f'Available webui options: {", ".join(WEBUI_LIST)}')
        return None, None, None

    if not arg2:
        print(f'{ERROR}: CivitAI API key is missing.')
        return None, None, None
    if re.search(r'\s+', arg2):
        print(f'{ERROR}: CivitAI API key contains spaces "{arg2}" - not allowed.')
        return None, None, None
    if len(arg2) < 32:
        print(f'{ERROR}: CivitAI API key must be at least 32 characters long.')
        return None, None, None

    if not arg3 or re.search(r'\s+', arg3):
        arg3 = ''

    selected_ui = next(option for option in WEBUI_LIST if arg1.lower() == option.lower())
    return selected_ui, arg2, arg3

# ---------------
# Rutas en Drive
# ---------------

def init_drive_paths():
    # Base en Drive
    DRIVE_BASE = Path('/content/drive/MyDrive/BigFutureWUI').resolve()
    DRIVE_BASE.mkdir(parents=True, exist_ok=True)

    # Estructura propuesta
    paths = {
        'BASE': DRIVE_BASE,
        'HOME': DRIVE_BASE,                     # Raíz de trabajo (repos, webuis)
        'TMP': DRIVE_BASE / 'temp',             # temporales
        'ENV': DRIVE_BASE / 'env',              # entorno “portable” (Python/Torch)
        'SRC': DRIVE_BASE / 'src',              # scripts propios, marcadores
        'BIN': DRIVE_BASE / 'bin',              # binarios (ngrok, zrok, cloudflared)
        'IPY': DRIVE_BASE / '.ipython_startup', # scripts de inicio que se ejecutan manualmente
        'ASD': DRIVE_BASE / 'asd',              # carpeta auxiliar para listas de extensiones/nodes
        'MODELS': DRIVE_BASE / 'models_cache'   # cachés compartidas/modelos (para enlaces simbólicos)
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

# --------------------
# Descarga de Python portable a Drive
# --------------------

def getPython():
    # Elegir versión y URLs según UI
    v = '3.11' if webui == 'Forge-Classic' else '3.10'
    BIN = str(PY / 'bin')
    PKG = str(PY / f'lib/python{v}/site-packages')

    if webui in ['ComfyUI', 'SwarmUI']:
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-ComfyUI-SwarmUI-Python310-Torch260-cu124.tar.lz4'
    elif webui == 'Forge-Classic':
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-FC-Python311-Torch260-cu124.tar.lz4'
    else:
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-Python310-Torch260-cu124.tar.lz4'

    fn = Path(url).name

    # Extraer en ENV (Drive)
    CD(str(ENV.parent))
    print(f"\n{ARROW} installing Python Portable {'3.11.13' if webui == 'Forge-Classic' else '3.10.15'} into {ENV}")

    SyS('sudo apt-get -qq -y install aria2 pv lz4 >/dev/null 2>&1')

    aria = f'aria2c --console-log-level=error --stderr=true -c -x16 -s16 -k1M -j5 {url} -o {fn}'
    p = subprocess.Popen(shlex.split(aria), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()

    # Extraer en la carpeta ENV
    ENV.mkdir(parents=True, exist_ok=True)
    CD(str(ENV))
    SyS(f'pv ../{fn} | lz4 -d | tar -xf -')
    Path(ENV.parent / fn).unlink()

    # Inyectar rutas
    sys.path.insert(0, PKG)
    if BIN not in iRON.get('PATH', ''):
        iRON['PATH'] = BIN + ':' + iRON.get('PATH', '')
    if PKG not in iRON.get('PYTHONPATH', ''):
        iRON['PYTHONPATH'] = PKG + ':' + iRON.get('PYTHONPATH', '')

def marking(p, n, u):
    # Persistente en Drive
    t = p / n
    v = {'ui': u, 'launch_args': '', 'tunnel': ''}

    if not t.exists():
        t.write_text(json.dumps(v, indent=4))

    d = json.loads(t.read_text())
    d.update(v)
    t.write_text(json.dumps(d, indent=4))

def key_inject(C, H):
    # Inserta claves en script auxiliar persistente en Drive
    p = Path(nenen)
    v = p.read_text()
    v = v.replace("TOKET = ''", f"TOKET = '{C}'")
    v = v.replace("TOBRUT = ''", f"TOBRUT = '{H}'")
    p.write_text(v)

def ensure_path_prepend(p: Path, var: str = 'PATH'):
    pv = str(p)
    if pv not in iRON.get(var, ''):
        iRON[var] = pv + ':' + iRON.get(var, '')

def install_tunnel():
    # Instala binarios en Drive (/bin) y los agrega al PATH
    BIN.mkdir(parents=True, exist_ok=True)
    ensure_path_prepend(BIN, 'PATH')

    # cloudflared
    cl_path = BIN / 'cloudflared'
    SyS(f'wget -qO {cl_path} https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64')
    SyS(f'chmod +x {cl_path}')

    bins = {
        'zrok': {
            'bin': BIN / 'zrok',
            'url': 'https://github.com/openziti/zrok/releases/download/v1.0.6/zrok_1.0.6_linux_amd64.tar.gz',
            'inner': None  # el tar ya contiene bin ejecutable con nombre zrok
        },
        'ngrok': {
            'bin': BIN / 'ngrok',
            'url': 'https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz',
            'inner': None
        }
    }

    for n, b in bins.items():
        if b['bin'].exists():
            b['bin'].unlink()

        url = b['url']
        name = Path(url).name
        tmp_archive = TMP / name
        SyS(f'wget -qO {tmp_archive} {url}')
        SyS(f'tar -xzf {tmp_archive} -C {BIN}')
        SyS(f'rm -f {tmp_archive}')

def sym_link(U, M):
    # Enlaces simbólicos para que los caches/modelos vivan en Drive (MODELS)
    configs = {
        'A1111': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'} {TMP}/*"
            ],
            'links': [
                (MODELS / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (MODELS / 'lora', M / 'Lora/tmp_lora'),
                (MODELS / 'controlnet', M / 'ControlNet')
            ]
        },

        'Forge': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}",
                f"rm -rf {M / 'svd'} {M / 'z123'} {M / 'clip'} {M / 'clip_vision'} {M / 'diffusers'}",
                f"rm -rf {M / 'diffusion_models'} {M / 'text_encoder'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (MODELS / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (MODELS / 'lora', M / 'Lora/tmp_lora'),
                (MODELS / 'controlnet', M / 'ControlNet'),
                (MODELS / 'z123', M / 'z123'),
                (MODELS / 'svd', M / 'svd'),
                (MODELS / 'clip', M / 'clip'),
                (MODELS / 'clip_vision', M / 'clip_vision'),
                (MODELS / 'diffusers', M / 'diffusers'),
                (MODELS / 'diffusion_models', M / 'diffusion_models'),
                (MODELS / 'text_encoders', M / 'text_encoder'),
                (MODELS / 'unet', M / 'unet')
            ]
        },

        'ReForge': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}",
                f"rm -rf {M / 'svd'} {M / 'z123'} {TMP}/*"
            ],
            'links': [
                (MODELS / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (MODELS / 'lora', M / 'Lora/tmp_lora'),
                (MODELS / 'controlnet', M / 'ControlNet'),
                (MODELS / 'z123', M / 'z123'),
                (MODELS / 'svd', M / 'svd')
            ]
        },

        'Forge-Classic': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}"
            ],
            'links': [
                (MODELS / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (MODELS / 'lora', M / 'Lora/tmp_lora'),
                (MODELS / 'controlnet', M / 'ControlNet')
            ]
        },

        'ComfyUI': {
            'sym': [
                f"rm -rf {M / 'checkpoints/tmp_ckpt'} {M / 'loras/tmp_lora'} {M / 'controlnet'}",
                f"rm -rf {M / 'clip'} {M / 'clip_vision'} {M / 'diffusers'} {M / 'diffusion_models'}",
                f"rm -rf {M / 'text_encoders'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (MODELS / 'ckpt', M / 'checkpoints/tmp_ckpt'),
                (MODELS / 'lora', M / 'loras/tmp_lora'),
                (MODELS / 'controlnet', M / 'controlnet'),
                (MODELS / 'clip', M / 'clip'),
                (MODELS / 'clip_vision', M / 'clip_vision'),
                (MODELS / 'diffusers', M / 'diffusers'),
                (MODELS / 'diffusion_models', M / 'diffusion_models'),
                (MODELS / 'text_encoders', M / 'text_encoders'),
                (MODELS / 'unet', M / 'unet')
            ]
        },

        'SwarmUI': {
            'sym': [
                f"rm -rf {M / 'Stable-Diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'controlnet'}",
                f"rm -rf {M / 'clip'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (MODELS / 'ckpt', M / 'Stable-Diffusion/tmp_ckpt'),
                (MODELS / 'lora', M / 'Lora/tmp_lora'),
                (MODELS / 'controlnet', M / 'controlnet'),
                (MODELS / 'clip', M / 'clip'),
                (MODELS / 'unet', M / 'unet')
            ]
        }
    }

    cfg = configs.get(U)
    [SyS(f'{cmd}') for cmd in cfg['sym']]
    if U not in ['ComfyUI', 'SwarmUI']:
        [(M / d).mkdir(parents=True, exist_ok=True) for d in ['Lora', 'ESRGAN']]
    [SyS(f'ln -s {src} {tg}')]  # noqa (no-op, se reemplaza abajo)
    for src, tg in cfg['links']:
        tg.parent.mkdir(parents=True, exist_ok=True)
        if tg.exists() or tg.is_symlink():
            SyS(f'rm -rf {tg}')
        SyS(f'ln -s {src} {tg}')

def webui_req(U, W, M):
    CD(W)

    if U != 'SwarmUI':
        pull(f'https://github.com/gutris1/segsmaker {U.lower()} {W}')
    else:
        M.mkdir(parents=True, exist_ok=True)
        for sub in ['Stable-Diffusion', 'Lora', 'Embeddings', 'VAE', 'upscale_models']:
            (M / sub).mkdir(parents=True, exist_ok=True)

        download(f'https://dot.net/v1/dotnet-install.sh {W}')
        dotnet = W / 'dotnet-install.sh'
        dotnet.chmod(0o755)
        SyS('bash ./dotnet-install.sh --channel 8.0')

    sym_link(U, M)
    install_tunnel()

    # Todos los auxiliares y upscalers se depositan en Drive
    scripts = [
        f'https://github.com/gutris1/segsmaker/raw/main/script/controlnet.py {W}/asd',
        f'https://github.com/gutris1/segsmaker/raw/main/script/KC/segsmaker.py {W}'
    ]

    u = M / 'upscale_models' if U in ['ComfyUI', 'SwarmUI'] else M / 'ESRGAN'

    upscalers = [
        f'https://huggingface.co/gutris1/webui/resolve/main/misc/4x-UltraSharp.pth {u}',
        f'https://huggingface.co/gutris1/webui/resolve/main/misc/4x-AnimeSharp.pth {u}',
        f'https://huggingface.co/gutris1/webui/resolve/main/misc/4x_NMKD-Superscale-SP_178000_G.pth {u}',
        f'https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth {u}',
        f'https://huggingface.co/gutris1/webui/resolve/main/misc/4x_RealisticRescaler_100000_G.pth {u}',
        f'https://huggingface.co/gutris1/webui/resolve/main/misc/8x_RealESRGAN.pth {u}',
        f'https://huggingface.co/gutris1/webui/resolve/main/misc/4x_foolhardy_Remacri.pth {u}',
        f'https://huggingface.co/subby2006/NMKD-YandereNeoXL/resolve/main/4x_NMKD-YandereNeoXL_200k.pth {u}',
        f'https://huggingface.co/subby2006/NMKD-UltraYandere/resolve/main/4x_NMKD-UltraYandere_300k.pth {u}'
    ]

    line = scripts + upscalers
    for item in line: download(item)

    if U not in ['SwarmUI', 'ComfyUI']:
        e = 'jpg' if U == 'Forge-Classic' else 'png'
        SyS(f'rm -f {W}/html/card-no-preview.{e}')

        for ass in [
            f'https://huggingface.co/gutris1/webui/resolve/main/misc/card-no-preview.png {W}/html card-no-preview.{e}',
            f'https://github.com/gutris1/segsmaker/raw/main/config/NoCrypt_miku.json {W}/tmp/gradio_themes',
            f'https://github.com/gutris1/segsmaker/raw/main/config/user.css {W} user.css'
        ]: download(ass)

        if U != 'Forge':
            download(f'https://github.com/gutris1/segsmaker/raw/main/config/config.json {W} config.json')

def webui_extension(U, W, M):
    EXT = W / 'custom_nodes' if U == 'ComfyUI' else W / 'extensions'
    CD(EXT)

    if U == 'ComfyUI':
        say('<br><b>【{red} Installing Custom Nodes{d} 】{red}</b>')
        clone(str(W / 'asd/custom_nodes.txt'))
        print()

        for faces in [
            f'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth {M}/facerestore_models',
            f'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth {M}/facerestore_models'
        ]: download(faces)

    else:
        say('<br><b>【{red} Installing Extensions{d} 】{red}</b>')
        clone(str(W / 'asd/extension.txt'))
        if ENVNAME == 'Kaggle':  # no-Drive path, pero lo dejamos por compatibilidad
            clone('https://github.com/gutris1/sd-image-encryption')

def webui_installation(U, W):
    M = W / 'Models' if U == 'SwarmUI' else W / 'models'
    E = M / 'Embeddings' if U == 'SwarmUI' else (M / 'embeddings' if U in ['Forge-Classic', 'ComfyUI'] else W / 'embeddings')
    V = M / 'vae' if U == 'ComfyUI' else M / 'VAE'

    webui_req(U, W, M)

    extras = [
        f'https://huggingface.co/gutris1/webui/resolve/main/misc/embeddingsXL.zip {W}',
        f'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors {V} sdxl_vae.safetensors'
    ]

    for i in extras: download(i)
    SyS(f"unzip -qo {W / 'embeddingsXL.zip'} -d {E} && rm {W / 'embeddingsXL.zip'}")

    if U != 'SwarmUI':
        webui_extension(U, W, M)

def webui_selection(ui):
    with output:
        output.clear_output(wait=True)

        if ui in REPO:
            (WEBUI, repo) = (HOME / ui, REPO[ui])
        say(f'<b>【{{red}} Installing {WEBUI.name} in Drive{{d}} 】{{red}}</b>')
        clone(repo)

        webui_installation(ui, WEBUI)

        with loading:
            loading.clear_output(wait=True)
            say('<br><b>【{red} Done{d} 】{red}</b>')
            tempe()
            CD(HOME)

def webui_installer():
    CD(HOME)
    ui = (json.load(MARKED.open('r')) if MARKED.exists() else {}).get('ui')
    WEBUI = HOME / ui if ui else None

    if WEBUI is not None and WEBUI.exists():
        git_dir = WEBUI / '.git'
        if git_dir.exists():
            CD(WEBUI)
            with output:
                output.clear_output(wait=True)
                if ui in ['A1111', 'ComfyUI', 'SwarmUI']:
                    SyS('git pull origin master')
                elif ui in ['Forge', 'ReForge']:
                    SyS('git pull origin main')
                elif ui == 'Forge-Classic':
                    SyS('git pull origin classic')
                with loading: loading.clear_output()
    else:
        try:
            webui_selection(webui)
        except KeyboardInterrupt:
            with loading: loading.clear_output()
            with output: print('\nCanceled.')
        except Exception as e:
            with loading: loading.clear_output()
            with output: print(f'\n{ERROR}: {e}')

def notebook_scripts():
    # Guardamos todos los scripts auxiliares en Drive (IPY)
    z = [
        (STR / '00-startup.py', f'wget -qO {STR}/00-startup.py https://github.com/gutris1/segsmaker/raw/main/script/KC/00-startup.py'),
        (nenen, f'wget -qO {nenen} https://github.com/gutris1/segsmaker/raw/main/script/nenen88.py'),
        (melon, f'wget -qO {melon} https://github.com/gutris1/segsmaker/raw/main/script/melon00.py'),
        (STR / 'cupang.py', f'wget -qO {STR}/cupang.py https://github.com/gutris1/segsmaker/raw/main/script/cupang.py'),
        (MRK, f'wget -qO {MRK} https://github.com/gutris1/segsmaker/raw/main/script/marking.py')
    ]

    [SyS(y) for x, y in z if not Path(x).exists()]

    # Persistimos variables de entorno/paths en Drive
    j = {'ENVNAME': ENVNAME, 'HOMEPATH': HOME, 'TEMPPATH': TMP, 'BASEPATH': Path(ENVBASE)}
    text = '\n'.join(f"{k} = '{v}'" for k, v in j.items())
    Path(KANDANG).write_text(text)

    key_inject(civitai_key, hf_read_token)
    marking(SRC, MARKED, webui)
    sys.path.append(str(STR))

    # Ejecutamos desde Drive (no dependemos de /root/.ipython)
    for scripts in [nenen, melon, KANDANG, MRK]:
        get_ipython().run_line_magic('run', str(scripts))

# -----------------------
# Punto de entrada
# -----------------------

# Detectar entorno
ENVNAME, ENVBASE, ENVHOME = getENV()

if not ENVNAME or ENVNAME != 'Colab':
    print('Este script está diseñado para Google Colab con Google Drive.\nExiting.')
    sys.exit()

# Montar Drive y preparar rutas
if not mount_drive_if_needed():
    print('No se pudo montar Google Drive. Exiting.')
    sys.exit()

# Colores / UI
RESET = '\033[0m'
RED = '\033[31m'
PURPLE = '\033[38;5;135m'
ORANGE = '\033[38;5;208m'
ARROW = f'{ORANGE}▶{RESET}'
ERROR = f'{PURPLE}[{RESET}{RED}ERROR{RESET}{PURPLE}]{RESET}'
IMG = 'https://github.com/gutris1/segsmaker/raw/main/script/loading.png'

# Rutas base en Drive
PATHS = init_drive_paths()
BASE = PATHS['BASE']
HOME = PATHS['HOME']
TMP = PATHS['TMP']
ENV = PATHS['ENV']
SRC = PATHS['SRC']
BIN = PATHS['BIN']
IPY = PATHS['IPY']
ASD = PATHS['ASD']
MODELS = PATHS['MODELS']

# Variables derivadas, todas en Drive
PY = ENV                       # carpeta del entorno portable
MRK = SRC / 'marking.py'
KEY = SRC / 'api-key.json'
MARKED = SRC / 'marking.json'

STR = IPY
nenen = STR / 'nenen88.py'
melon = STR / 'melon00.py'
KANDANG = STR / 'KANDANG.py'

# Crear carpetas
for p in [TMP, SRC, STR, BIN, MODELS, ASD]:
    p.mkdir(parents=True, exist_ok=True)

# Widgets de salida
output = widgets.Output()
loading = widgets.Output()

# Argumentos
webui, civitai_key, hf_read_token = getArgs()
if civitai_key is None:
    sys.exit()

# Mostrar cargador
display(output, loading)
with loading:
    display(Image(url=IMG))

# Asegurar entorno portable en Drive
with output:
    PY.exists() or getPython()

# Importar helpers (se descargarán a Drive) y continuar
notebook_scripts()

from nenen88 import clone, say, download, tempe, pull

# Añadir binarios de Drive al PATH por si nenen88/otros los invocan
ensure_path_prepend(BIN, 'PATH')

# Instalar/actualizar el UI seleccionado en Drive
webui_installer()