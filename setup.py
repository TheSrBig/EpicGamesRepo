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

def getENV():
    env = {
        'Colab': ('/content/drive/MyDrive/BigFutureWUI', '/content/drive/MyDrive/BigFutureWUI', 'COLAB_JUPYTER_TOKEN'),
        'Kaggle': ('/kaggle', '/kaggle/working', 'KAGGLE_DATA_PROXY_TOKEN')
    }
    for name, (base, home, var) in env.items():
        if var in iRON:
            # Asegurar que el directorio de Drive existe
            if name == 'Colab':
                os.makedirs(base, exist_ok=True)
                os.makedirs(home, exist_ok=True)
            return name, base, home
    return None, None, None

def getArgs():
    parser = argparse.ArgumentParser(description='WebUI Installer Script for Kaggle and Google Colab with Drive support')
    parser.add_argument('--webui', required=True, help='available webui: A1111, Forge, ReForge, Forge-Classic, ComfyUI, SwarmUI')
    parser.add_argument('--civitai_key', required=True, help='your CivitAI API key')
    parser.add_argument('--hf_read_token', default=None, help='your Huggingface READ Token (optional)')

    args, unknown = parser.parse_known_args()

    arg1 = args.webui.lower()
    arg2 = args.civitai_key.strip()
    arg3 = args.hf_read_token.strip() if args.hf_read_token else ''

    if not any(arg1 == option.lower() for option in WEBUI_LIST):
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

    if not arg3: arg3 = ''
    if re.search(r'\s+', arg3): arg3 = ''

    selected_ui = next(option for option in WEBUI_LIST if arg1 == option.lower())
    return selected_ui, arg2, arg3

def getPython():
    v = '3.11' if webui == 'Forge-Classic' else '3.10'
    
    # Usar Python del sistema de Colab pero crear directorios en Drive
    BIN = '/usr/bin'  # Usar Python del sistema
    PKG_SYSTEM = f'/usr/local/lib/python{v}/dist-packages'
    
    # Crear directorio para packages adicionales en Drive
    PKG_DRIVE = str(ENVBASE / f'python_packages/lib/python{v}/site-packages')
    os.makedirs(PKG_DRIVE, exist_ok=True)

    if webui in ['ComfyUI', 'SwarmUI']:
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-ComfyUI-SwarmUI-Python310-Torch260-cu124.tar.lz4'
    elif webui == 'Forge-Classic':
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-FC-Python311-Torch260-cu124.tar.lz4'
    else:
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-Python310-Torch260-cu124.tar.lz4'

    fn = Path(url).name

    CD(ENVBASE)
    print(f"\n{ARROW} Setting up Python environment in Drive")

    # Instalar herramientas necesarias
    SyS('apt-get -qq -y install aria2 pv lz4 >/dev/null 2>&1')

    # Descargar y extraer paquetes adicionales en Drive
    aria = f'aria2c --console-log-level=error --stderr=true -c -x16 -s16 -k1M -j5 {url} -o {fn}'
    p = subprocess.Popen(shlex.split(aria), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()

    SyS(f'pv {fn} | lz4 -d | tar -xf -')
    Path(fn).unlink()

    # Añadir tanto el sistema como Drive al path de Python
    sys.path.insert(0, PKG_DRIVE)
    sys.path.insert(0, PKG_SYSTEM)
    
    if PKG_DRIVE not in iRON.get('PYTHONPATH', ''): 
        iRON['PYTHONPATH'] = PKG_DRIVE + ':' + iRON.get('PYTHONPATH', '')

    # Instalar ipywidgets en Drive si es necesario
    if ENVNAME == 'Colab':
        SyS(f'pip install --target {PKG_DRIVE} ipywidgets jupyterlab_widgets --upgrade --quiet')

def marking(p, n, u):
    t = p / n
    v = {'ui': u, 'launch_args': '', 'tunnel': ''}

    if not t.exists(): t.write_text(json.dumps(v, indent=4))

    d = json.loads(t.read_text())
    d.update(v)
    t.write_text(json.dumps(d, indent=4))

def key_inject(C, H):
    p = Path(nenen)
    v = p.read_text()
    v = v.replace("TOKET = ''", f"TOKET = '{C}'")
    v = v.replace("TOBRUT = ''", f"TOBRUT = '{H}'")
    p.write_text(v)

def install_tunnel():
    # Instalar túneles en directorio temporal del sistema (no en Drive)
    tunnel_dir = Path('/tmp/tunnels')
    tunnel_dir.mkdir(exist_ok=True)
    
    SyS(f'wget -qO {tunnel_dir}/cl https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64')
    SyS(f'chmod +x {tunnel_dir}/cl')
    
    # Crear enlaces simbólicos en /usr/bin
    SyS(f'ln -sf {tunnel_dir}/cl /usr/bin/cl')

    bins = {
        'zrok': {
            'bin': tunnel_dir / 'zrok',
            'url': 'https://github.com/openziti/zrok/releases/download/v1.0.6/zrok_1.0.6_linux_amd64.tar.gz'
        },
        'ngrok': {
            'bin': tunnel_dir / 'ngrok',
            'url': 'https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz'
        }
    }

    for n, b in bins.items():
        if b['bin'].exists(): b['bin'].unlink()

        url = b['url']
        name = Path(url).name

        SyS(f'wget -qO {tunnel_dir}/{name} {url}')
        SyS(f'tar -xzf {tunnel_dir}/{name} -C {tunnel_dir}')
        SyS(f'rm -f {tunnel_dir}/{name}')
        SyS(f'ln -sf {tunnel_dir}/{n} /usr/bin/{n}')

def sym_link(U, M):
    configs = {
        'A1111': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet')
            ]
        },

        'Forge': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}",
                f"rm -rf {M / 'svd'} {M / 'z123'} {M / 'clip'} {M / 'clip_vision'} {M / 'diffusers'}",
                f"rm -rf {M / 'diffusion_models'} {M / 'text_encoder'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet'),
                (TMP / 'z123', M / 'z123'),
                (TMP / 'svd', M / 'svd'),
                (TMP / 'clip', M / 'clip'),
                (TMP / 'clip_vision', M / 'clip_vision'),
                (TMP / 'diffusers', M / 'diffusers'),
                (TMP / 'diffusion_models', M / 'diffusion_models'),
                (TMP / 'text_encoders', M / 'text_encoder'),
                (TMP / 'unet', M / 'unet')
            ]
        },

        'ReForge': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}",
                f"rm -rf {M / 'svd'} {M / 'z123'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet'),
                (TMP / 'z123', M / 'z123'),
                (TMP / 'svd', M / 'svd')
            ]
        },

        'Forge-Classic': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet')
            ]
        },

        'ComfyUI': {
            'sym': [
                f"rm -rf {M / 'checkpoints/tmp_ckpt'} {M / 'loras/tmp_lora'} {M / 'controlnet'}",
                f"rm -rf {M / 'clip'} {M / 'clip_vision'} {M / 'diffusers'} {M / 'diffusion_models'}",
                f"rm -rf {M / 'text_encoders'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'checkpoints/tmp_ckpt'),
                (TMP / 'lora', M / 'loras/tmp_lora'),
                (TMP / 'controlnet', M / 'controlnet'),
                (TMP / 'clip', M / 'clip'),
                (TMP / 'clip_vision', M / 'clip_vision'),
                (TMP / 'diffusers', M / 'diffusers'),
                (TMP / 'diffusion_models', M / 'diffusion_models'),
                (TMP / 'text_encoders', M / 'text_encoders'),
                (TMP / 'unet', M / 'unet')
            ]
        },

        'SwarmUI': {
            'sym': [
                f"rm -rf {M / 'Stable-Diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'controlnet'}",
                f"rm -rf {M / 'clip'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-Diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'controlnet'),
                (TMP / 'clip', M / 'clip'),
                (TMP / 'unet', M / 'unet')
            ]
        }
    }

    cfg = configs.get(U)
    [SyS(f'{cmd}') for cmd in cfg['sym']]
    if U not in ['ComfyUI', 'SwarmUI']: [(M / d).mkdir(parents=True, exist_ok=True) for d in ['Lora', 'ESRGAN']]
    [SyS(f'ln -s {src} {tg}') for src, tg in cfg['links']]

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

        if U != 'Forge': download(f'https://github.com/gutris1/segsmaker/raw/main/config/config.json {W} config.json')

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
        if ENVNAME == 'Kaggle': clone('https://github.com/gutris1/sd-image-encryption')

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

    if U != 'SwarmUI': webui_extension(U, W, M)

def webui_selection(ui):
    with output:
        output.clear_output(wait=True)

        if ui in REPO: (WEBUI, repo) = (HOME / ui, REPO[ui])
        say(f'<b>【{{red}} Installing {WEBUI.name} in Drive{{d}} 】{{red}}</b>')
        clone(repo)

        webui_installation(ui, WEBUI)

        with loading:
            loading.clear_output(wait=True)
            say('<br><b>【{red} Done - Everything saved in Drive{d} 】{red}</b>')
            
            # Configurar variables de entorno para las celdas del notebook
            iRON['WebUI'] = str(WEBUI)
            iRON['Extensions'] = str(WEBUI / ('custom_nodes' if ui == 'ComfyUI' else 'extensions'))
            
            M = WEBUI / 'Models' if ui == 'SwarmUI' else WEBUI / 'models'
            E = M / 'Embeddings' if ui == 'SwarmUI' else (M / 'embeddings' if ui in ['Forge-Classic', 'ComfyUI'] else WEBUI / 'embeddings')
            V = M / 'vae' if ui == 'ComfyUI' else M / 'VAE'
            U = M / 'upscale_models' if ui in ['ComfyUI', 'SwarmUI'] else M / 'ESRGAN'
            
            iRON['VAE'] = str(V)
            iRON['Embeddings'] = str(E)
            iRON['Upscalers'] = str(U)
            iRON['CKPT'] = str(M / ('checkpoints' if ui == 'ComfyUI' else ('Stable-Diffusion' if ui == 'SwarmUI' else 'Stable-diffusion')))
            iRON['LORA'] = str(M / ('loras' if ui == 'ComfyUI' else 'Lora'))
            
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
                say(f'<b>【{{red}} Updating {ui} in Drive{{d}} 】{{red}}</b>')
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
    # Crear directorios necesarios en Drive
    STR.mkdir(parents=True, exist_ok=True)
    
    z = [
        (STR / '00-startup.py', f'wget -qO {STR}/00-startup.py https://github.com/gutris1/segsmaker/raw/main/script/KC/00-startup.py'),
        (nenen, f'wget -qO {nenen} https://github.com/gutris1/segsmaker/raw/main/script/nenen88.py'),
        (melon, f'wget -qO {melon} https://github.com/gutris1/segsmaker/raw/main/script/melon00.py'),
        (STR / 'cupang.py', f'wget -qO {STR}/cupang.py https://github.com/gutris1/segsmaker/raw/main/script/cupang.py'),
        (MRK, f'wget -qO {MRK} https://github.com/gutris1/segsmaker/raw/main/script/marking.py')
    ]

    [SyS(y) for x, y in z if not Path(x).exists()]

    j = {'ENVNAME': ENVNAME, 'HOMEPATH': HOME, 'TEMPPATH': TMP, 'BASEPATH': Path(ENVBASE)}
    text = '\n'.join(f"{k} = '{v}'" for k, v in j.items())
    Path(KANDANG).write_text(text)

    key_inject(civitai_key, hf_read_token)
    marking(SRC, MARKED, webui)
    sys.path.append(str(STR))

    for scripts in [nenen, melon, KANDANG, MRK]: get_ipython().run_line_magic('run', str(scripts))

ENVNAME, ENVBASE, ENVHOME = getENV()

if not ENVNAME:
    print('You are not in Kaggle or Google Colab.\nExiting.')
    sys.exit()

RESET = '\033[0m'
RED = '\033[31m'
PURPLE = '\033[38;5;135m'
ORANGE = '\033[38;5;208m'
ARROW = f'{ORANGE}▶{RESET}'
ERROR = f'{PURPLE}[{RESET}{RED}ERROR{RESET}{PURPLE}]{RESET}'
IMG = 'https://github.com/gutris1/segsmaker/raw/main/script/loading.png'

# Cambiar todas las rutas para usar Drive
HOME = Path(ENVHOME)
TMP = HOME / 'temp'  # Temporal también en Drive para persistencia

PY = Path('/GUTRIS1')  # Mantener Python portable en sistema
SRC = HOME / 'gutris1'
MRK = SRC / 'marking.py'
KEY = SRC / 'api-key.json'
MARKED = SRC / 'marking.json'

USR = Path('/usr/bin')
STR = HOME / '.ipython/profile_default/startup'  # Scripts en Drive
nenen = STR / 'nenen88.py'
melon = STR / 'melon00.py'
KANDANG = STR / 'KANDANG.py'

TMP.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

output = widgets.Output()
loading = widgets.Output()

webui, civitai_key, hf_read_token = getArgs()
if civitai_key is None: sys.exit()

display(output, loading)
with loading: display(Image(url=IMG))
with output: 
    print(f"{ARROW} Using Drive directory: {HOME}")
    PY.exists() or getPython()
notebook_scripts()

from nenen88 import clone, say, download, tempe, pull
webui_installer()
