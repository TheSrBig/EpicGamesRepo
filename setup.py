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
        'Colab': ('/content', '/content', 'COLAB_JUPYTER_TOKEN'),
        'Kaggle': ('/kaggle', '/kaggle/working', 'KAGGLE_DATA_PROXY_TOKEN')
    }
    for name, (base, home, var) in env.items():
        if var in iRON:
            return name, base, home
    return None, None, None

def getArgs():
    parser = argparse.ArgumentParser(description='WebUI Installer Script for Kaggle and Google Colab with Drive Support')
    parser.add_argument('--webui', required=True, help='available webui: A1111, Forge, ReForge, Forge-Classic, ComfyUI, SwarmUI')
    parser.add_argument('--civitai_key', required=True, help='your CivitAI API key')
    parser.add_argument('--hf_read_token', default=None, help='your Huggingface READ Token (optional)')
    parser.add_argument('--use_drive', action='store_true', help='Install in Google Drive instead of local storage')
    parser.add_argument('--drive_path', default='/content/drive/MyDrive/BigFutureWUI', help='Path in Google Drive to install')

    args, unknown = parser.parse_known_args()

    arg1 = args.webui.lower()
    arg2 = args.civitai_key.strip()
    arg3 = args.hf_read_token.strip() if args.hf_read_token else ''

    if not any(arg1 == option.lower() for option in WEBUI_LIST):
        print(f'{ERROR}: invalid webui option: "{args.webui}"')
        print(f'Available webui options: {", ".join(WEBUI_LIST)}')
        return None, None, None, None, None

    if not arg2:
        print(f'{ERROR}: CivitAI API key is missing.')
        return None, None, None, None, None
    if re.search(r'\s+', arg2):
        print(f'{ERROR}: CivitAI API key contains spaces "{arg2}" - not allowed.')
        return None, None, None, None, None
    if len(arg2) < 32:
        print(f'{ERROR}: CivitAI API key must be at least 32 characters long.')
        return None, None, None, None, None

    if not arg3: arg3 = ''
    if re.search(r'\s+', arg3): arg3 = ''

    selected_ui = next(option for option in WEBUI_LIST if arg1 == option.lower())
    return selected_ui, arg2, arg3, args.use_drive, args.drive_path

def mount_drive():
    """Mount Google Drive if not already mounted"""
    drive_path = Path('/content/drive')
    if not drive_path.exists():
        print(f"{ARROW} Mounting Google Drive...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print(f"{ARROW} Google Drive mounted successfully")
        except ImportError:
            print(f"{ERROR}: Not running in Google Colab, cannot mount Drive")
            return False
        except Exception as e:
            print(f"{ERROR}: Failed to mount Google Drive: {e}")
            return False
    else:
        print(f"{ARROW} Google Drive already mounted")
    return True

def setup_drive_environment(drive_path):
    """Setup the Drive environment structure"""
    drive_base = Path(drive_path)
    drive_base.mkdir(parents=True, exist_ok=True)
    
    # Create necessary directories in Drive
    directories = [
        'webui',
        'models',
        'temp',
        'python_env',
        'scripts'
    ]
    
    for dir_name in directories:
        (drive_base / dir_name).mkdir(parents=True, exist_ok=True)
    
    return drive_base

def getPython():
    v = '3.11' if webui == 'Forge-Classic' else '3.10'
    
    if use_drive:
        # Install Python in Drive
        PY_DRIVE = DRIVE_BASE / 'python_env'
        BIN = str(PY_DRIVE / 'bin')
        PKG = str(PY_DRIVE / f'lib/python{v}/site-packages')
        PY_LOCAL = PY_DRIVE
    else:
        # Original local installation
        BIN = str(PY / 'bin')
        PKG = str(PY / f'lib/python{v}/site-packages')
        PY_LOCAL = PY

    if webui in ['ComfyUI', 'SwarmUI']:
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-ComfyUI-SwarmUI-Python310-Torch260-cu124.tar.lz4'
    elif webui == 'Forge-Classic':
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-FC-Python311-Torch260-cu124.tar.lz4'
    else:
        url = 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-Python310-Torch260-cu124.tar.lz4'

    fn = Path(url).name

    if use_drive:
        CD(DRIVE_BASE)
    else:
        CD(Path(ENVBASE).parent)
    
    print(f"\n{ARROW} installing Python Portable {'3.11.13' if webui == 'Forge-Classic' else '3.10.15'}")
    if use_drive:
        print(f"{ARROW} Installing in Google Drive: {PY_LOCAL}")

    SyS('sudo apt-get -qq -y install aria2 pv lz4 >/dev/null 2>&1')

    aria = f'aria2c --console-log-level=error --stderr=true -c -x16 -s16 -k1M -j5 {url} -o {fn}'
    p = subprocess.Popen(shlex.split(aria), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()

    if use_drive:
        SyS(f'pv {fn} | lz4 -d | tar -xf - -C {DRIVE_BASE / "python_env"}')
    else:
        SyS(f'pv {fn} | lz4 -d | tar -xf -')
    
    Path(fn).unlink()

    sys.path.insert(0, PKG)
    if BIN not in iRON['PATH']: iRON['PATH'] = BIN + ':' + iRON['PATH']
    if PKG not in iRON['PYTHONPATH']: iRON['PYTHONPATH'] = PKG + ':' + iRON['PYTHONPATH']

    if ENVNAME == 'Kaggle':
        for cmd in [
            'pip install ipywidgets jupyterlab_widgets --upgrade',
            'rm -f /usr/lib/python3.10/sitecustomize.py'
        ]: SyS(f'{cmd} >/dev/null 2>&1')

def marking(p, n, u):
    t = p / n
    v = {'ui': u, 'launch_args': '', 'tunnel': '', 'use_drive': use_drive, 'drive_path': str(DRIVE_BASE) if use_drive else ''}

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
    tunnel_dir = DRIVE_BASE / 'scripts' if use_drive else USR
    tunnel_dir.mkdir(parents=True, exist_ok=True)
    
    SyS(f'wget -qO {tunnel_dir}/cl https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64')
    SyS(f'chmod +x {tunnel_dir}/cl')

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

        SyS(f'wget -qO {name} {url}')
        SyS(f'tar -xzf {name} -C {tunnel_dir}')
        SyS(f'rm -f {name}')
    
    # Add tunnel directory to PATH if using Drive
    if use_drive:
        tunnel_bin = str(tunnel_dir)
        if tunnel_bin not in iRON['PATH']: 
            iRON['PATH'] = tunnel_bin + ':' + iRON['PATH']

def sym_link(U, M):
    if use_drive:
        # Use Drive temp directory
        temp_dir = DRIVE_BASE / 'temp'
    else:
        temp_dir = TMP
    
    temp_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        'A1111': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'} {temp_dir}/*"
            ],
            'links': [
                (temp_dir / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (temp_dir / 'lora', M / 'Lora/tmp_lora'),
                (temp_dir / 'controlnet', M / 'ControlNet')
            ]
        },

        'Forge': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}",
                f"rm -rf {M / 'svd'} {M / 'z123'} {M / 'clip'} {M / 'clip_vision'} {M / 'diffusers'}",
                f"rm -rf {M / 'diffusion_models'} {M / 'text_encoder'} {M / 'unet'} {temp_dir}/*"
            ],
            'links': [
                (temp_dir / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (temp_dir / 'lora', M / 'Lora/tmp_lora'),
                (temp_dir / 'controlnet', M / 'ControlNet'),
                (temp_dir / 'z123', M / 'z123'),
                (temp_dir / 'svd', M / 'svd'),
                (temp_dir / 'clip', M / 'clip'),
                (temp_dir / 'clip_vision', M / 'clip_vision'),
                (temp_dir / 'diffusers', M / 'diffusers'),
                (temp_dir / 'diffusion_models', M / 'diffusion_models'),
                (temp_dir / 'text_encoders', M / 'text_encoder'),
                (temp_dir / 'unet', M / 'unet')
            ]
        },

        'ReForge': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}",
                f"rm -rf {M / 'svd'} {M / 'z123'} {temp_dir}/*"
            ],
            'links': [
                (temp_dir / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (temp_dir / 'lora', M / 'Lora/tmp_lora'),
                (temp_dir / 'controlnet', M / 'ControlNet'),
                (temp_dir / 'z123', M / 'z123'),
                (temp_dir / 'svd', M / 'svd')
            ]
        },

        'Forge-Classic': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}"
            ],
            'links': [
                (temp_dir / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (temp_dir / 'lora', M / 'Lora/tmp_lora'),
                (temp_dir / 'controlnet', M / 'ControlNet')
            ]
        },

        'ComfyUI': {
            'sym': [
                f"rm -rf {M / 'checkpoints/tmp_ckpt'} {M / 'loras/tmp_lora'} {M / 'controlnet'}",
                f"rm -rf {M / 'clip'} {M / 'clip_vision'} {M / 'diffusers'} {M / 'diffusion_models'}",
                f"rm -rf {M / 'text_encoders'} {M / 'unet'} {temp_dir}/*"
            ],
            'links': [
                (temp_dir / 'ckpt', M / 'checkpoints/tmp_ckpt'),
                (temp_dir / 'lora', M / 'loras/tmp_lora'),
                (temp_dir / 'controlnet', M / 'controlnet'),
                (temp_dir / 'clip', M / 'clip'),
                (temp_dir / 'clip_vision', M / 'clip_vision'),
                (temp_dir / 'diffusers', M / 'diffusers'),
                (temp_dir / 'diffusion_models', M / 'diffusion_models'),
                (temp_dir / 'text_encoders', M / 'text_encoders'),
                (temp_dir / 'unet', M / 'unet')
            ]
        },

        'SwarmUI': {
            'sym': [
                f"rm -rf {M / 'Stable-Diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'controlnet'}",
                f"rm -rf {M / 'clip'} {M / 'unet'} {temp_dir}/*"
            ],
            'links': [
                (temp_dir / 'ckpt', M / 'Stable-Diffusion/tmp_ckpt'),
                (temp_dir / 'lora', M / 'Lora/tmp_lora'),
                (temp_dir / 'controlnet', M / 'controlnet'),
                (temp_dir / 'clip', M / 'clip'),
                (temp_dir / 'unet', M / 'unet')
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

        webui_dir = DRIVE_BASE / 'webui' if use_drive else HOME
        WEBUI = webui_dir / ui
        
        if ui in REPO: repo = REPO[ui]
        say(f'<b>【{{red}} Installing {WEBUI.name}{{d}} 】{{red}}</b>')
        if use_drive:
            say(f'<b>【{{red}} Installing in Google Drive: {WEBUI}{{d}} 】{{red}}</b>')
        
        clone(repo)

        webui_installation(ui, WEBUI)

        with loading:
            loading.clear_output(wait=True)
            say('<br><b>【{red} Done{d} 】{red}</b>')
            tempe()
            CD(HOME)

def webui_installer():
    webui_dir = DRIVE_BASE / 'webui' if use_drive else HOME
    CD(webui_dir)
    
    ui = (json.load(MARKED.open('r')) if MARKED.exists() else {}).get('ui')
    WEBUI = webui_dir / ui if ui else None

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
    scripts_dir = DRIVE_BASE / 'scripts' if use_drive else STR
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    nenen_path = scripts_dir / 'nenen88.py'
    melon_path = scripts_dir / 'melon00.py'
    cupang_path = scripts_dir / 'cupang.py'
    startup_path = scripts_dir / '00-startup.py'
    
    z = [
        (startup_path, f'wget -qO {startup_path} https://github.com/gutris1/segsmaker/raw/main/script/KC/00-startup.py'),
        (nenen_path, f'wget -qO {nenen_path} https://github.com/gutris1/segsmaker/raw/main/script/nenen88.py'),
        (melon_path, f'wget -qO {melon_path} https://github.com/gutris1/segsmaker/raw/main/script/melon00.py'),
        (cupang_path, f'wget -qO {cupang_path} https://github.com/gutris1/segsmaker/raw/main/script/cupang.py'),
        (MRK, f'wget -qO {MRK} https://github.com/gutris1/segsmaker/raw/main/script/marking.py')
    ]

    [SyS(y) for x, y in z if not Path(x).exists()]

    base_path = DRIVE_BASE if use_drive else Path(ENVBASE)
    home_path = DRIVE_BASE / 'webui' if use_drive else HOME
    temp_path = DRIVE_BASE / 'temp' if use_drive else TMP
    
    j = {
        'ENVNAME': ENVNAME, 
        'HOMEPATH': home_path, 
        'TEMPPATH': temp_path, 
        'BASEPATH': base_path,
        'USE_DRIVE': use_drive,
        'DRIVE_PATH': str(DRIVE_BASE) if use_drive else ''
    }
    text = '\n'.join(f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}" for k, v in j.items())
    
    kandang_path = scripts_dir / 'KANDANG.py'
    kandang_path.write_text(text)

    key_inject(civitai_key, hf_read_token)
    marking(SRC, MARKED, webui)
    sys.path.append(str(scripts_dir))

    # Update global variables for script imports
    global nenen, melon, KANDANG
    nenen = nenen_path
    melon = melon_path
    KANDANG = kandang_path

    for scripts in [nenen, melon, KANDANG, MRK]: 
        get_ipython().run_line_magic('run', str(scripts))

# Initialize environment
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

# Parse arguments first to determine if using Drive
webui, civitai_key, hf_read_token, use_drive, drive_path = getArgs()
if civitai_key is None: sys.exit()

# Setup paths based on Drive usage
if use_drive:
    if not mount_drive():
        print(f"{ERROR}: Failed to mount Google Drive. Falling back to local installation.")
        use_drive = False

if use_drive:
    DRIVE_BASE = setup_drive_environment(drive_path)
    HOME = DRIVE_BASE / 'webui'
    TMP = DRIVE_BASE / 'temp'
    SRC = DRIVE_BASE / 'scripts' / 'gutris1'
else:
    HOME = Path(ENVHOME)
    TMP = Path(ENVBASE) / 'temp'
    SRC = HOME / 'gutris1'

# Create necessary directories
HOME.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

# Setup remaining paths
MRK = SRC / 'marking.py'
KEY = SRC / 'api-key.json'
MARKED = SRC / 'marking.json'

USR = Path('/usr/bin')
STR = Path('/root/.ipython/profile_default/startup')

# These will be updated in notebook_scripts() if using Drive
nenen = STR / 'nenen88.py'
melon = STR / 'melon00.py'
KANDANG = STR / 'KANDANG.py'

PY = Path('/GUTRIS1')

output = widgets.Output()
loading = widgets.Output()

display(output, loading)
with loading: display(Image(url=IMG))
with output: 
    if not (DRIVE_BASE / 'python_env' if use_drive else Path('/GUTRIS1')).exists():
        getPython()

notebook_scripts()

from nenen88 import clone, say, download, tempe, pull
webui_installer()
