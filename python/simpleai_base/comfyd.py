import subprocess
import os
import sys
import site
import torch
import gc
import time
import ldm_patched.modules.model_management as model_management
from . import comfyclient_pipeline, utils
from simpleai_base.simpleai_base import gen_entry_point_id

comfyd_process = None
comfyd_active = False
comfyd_args = [[]]


def _normalized_path(path):
    if not path:
        return None
    try:
        return os.path.normcase(os.path.abspath(path))
    except Exception:
        return os.path.normcase(str(path))


def _user_site_paths():
    paths = []
    try:
        user_sites = site.getusersitepackages()
        if isinstance(user_sites, str):
            paths.append(user_sites)
        else:
            paths.extend(user_sites)
    except Exception:
        pass
    for attr in ("USER_SITE", "USER_BASE"):
        value = getattr(site, attr, None)
        if value:
            paths.append(value)
    return {normalized for normalized in (_normalized_path(path) for path in paths) if normalized}


def _is_user_site_path(path, user_sites):
    normalized = _normalized_path(path)
    if not normalized:
        return False
    for user_site in user_sites:
        if normalized == user_site or normalized.startswith(user_site + os.sep):
            return True
    return False


def _filtered_pythonpath():
    user_sites = _user_site_paths()
    result = []
    seen = set()
    for path in sys.path:
        if not path or _is_user_site_path(path, user_sites):
            continue
        normalized = _normalized_path(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(path)
    return result


def _build_process_env():
    process_env = os.environ.copy()
    process_env["PYTHONNOUSERSITE"] = "1"
    process_env["PIP_USER"] = "0"
    process_env.pop("PYTHONUSERBASE", None)
    process_env["PYTHONPATH"] = os.pathsep.join(_filtered_pythonpath())
    return process_env


def is_running():
    global comfyd_process
    if 'comfyd_process' not in globals():
        return False
    if comfyd_process is None:
        return False
    process_code = comfyd_process.poll()
    if process_code is None:
        return True
    print(f"{utils.now_string()} [Comfyd] comfyd process status code: {process_code}")
    return False


def start(args_patch=[[]], force=False):
    global comfyd_process, comfyd_args
    if force:
        stop(force)
        time.sleep(1)

    if not is_running():
        backend_script = os.path.join(os.getcwd(), 'comfy/main.py')
        args_comfyd = [["--preview-method", "auto"], ["--port", "8187"], ["--disable-auto-launch"]]
        patches = []
        for patch in comfyd_args:
            if isinstance(patch, (list, tuple)) and len(patch) > 0:
                patches.append(list(patch))
        if isinstance(args_patch, (list, tuple)):
            for patch in args_patch:
                if isinstance(patch, (list, tuple)) and len(patch) > 0:
                    patches.append(list(patch))

        for patch in patches:
            found = False
            for i, sublist in enumerate(args_comfyd):
                if sublist and sublist[0] == patch[0]:
                    if len(sublist) > 1 and len(patch) > 1:
                        args_comfyd[i][1] = patch[1]
                    found = True
                    break
            if not found:
                args_comfyd.append(patch)
        arguments = [arg for sublist in args_comfyd for arg in sublist]
        process_env = _build_process_env()
        model_management.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        if not utils.echo_off:
            print(f'{utils.now_string()} [Comfyd] Ready to start with arguments: {arguments}, env: {process_env}')
        if 'comfyd_process' not in globals():
            globals()['comfyd_process'] = None
        comfyd_process = subprocess.Popen([sys.executable, "-s", backend_script] + arguments, env=process_env)
        comfyclient_pipeline.COMFYUI_ENDPOINT_PORT = [arg[1] for arg in args_comfyd if arg[0] == "--port"][0]
        comfyclient_pipeline.ws = None

    else:
        print(f"{utils.now_string()} [Comfyd] Comfyd is active!")
    return


def active(flag=False):
    global comfyd_active
    comfyd_active = flag
    if flag and not is_running():
        start()
    if not flag and is_running():
        stop()
    return


def finished():
    global comfyd_process
    if 'comfyd_process' not in globals():
        return
    if comfyd_process is None:
        return
    if comfyd_active:
        # free()
        gc.collect()
        print(f"{utils.now_string()} [Comfyd] Task finished!")
        return
    comfyclient_pipeline.ws = None
    free()
    gc.collect()
    print(f"{utils.now_string()} [Comfyd] Comfyd stopped!")


def stop(force=False):
    global comfyd_process
    if 'comfyd_process' not in globals():
        return
    if comfyd_process is None:
        return
    if comfyd_active and not force:
        free(all=True)
        gc.collect()
        print(f"{utils.now_string()} [Comfyd] Comfyd freeing!")
        return
    if is_running():
        comfyd_process.terminate()
        comfyd_process.wait()
    del comfyd_process
    comfyclient_pipeline.ws = None
    free()
    gc.collect()
    print(f"{utils.now_string()} [Comfyd] Comfyd stopped!")


def free(all=False):
    global comfyd_process
    if 'comfyd_process' not in globals():
        return
    if comfyd_process is None:
        return
    comfyclient_pipeline.free(all)
    return

def modify_variable(vars):
    comfyclient_pipeline.setvars(vars)
    return

def interrupt():
    global comfyd_process
    if 'comfyd_process' not in globals():
        return
    if comfyd_process is None:
        return
    comfyclient_pipeline.interrupt()
    return


def args_mapping(args_fooocus):
    args_comfy = []
    if "--gpu-device-id" in args_fooocus:
        args_comfy += [["--cuda-device", args_fooocus[args_fooocus.index("--gpu-device-id") + 1]]]
    if "--async-cuda-allocation" in args_fooocus:
        args_comfy += [["--cuda-malloc"]]
    if "--disable-async-cuda-allocation" in args_fooocus:
        args_comfy += [["--disable-cuda-malloc"]]
    if "--vae-in-cpu" in args_fooocus:
        args_comfy += [["--vae-in-cpu"]]
    if "--directml" in args_fooocus:
        args_comfy += [["--directml"]]
    if "--disable-xformers" in args_fooocus:
        args_comfy += [["--disable-xformers"]]
    if "--always-cpu" in args_fooocus:
        args_comfy += [["--cpu"]]
    if "--always-low-vram" in args_fooocus:
        args_comfy += [["--lowvram"]]
    if "--always-gpu" in args_fooocus:
        args_comfy += [["--gpu-only"]]
    if "--always-offload-from-vram" in args_fooocus:
        args_comfy += [["--disable-smart-memory"]]
    if not utils.echo_off:
        print(f'{utils.now_string()} [Comfyd] args_fooocus: {args_fooocus}\nargs_comfy: {args_comfy}')
    return args_comfy


def get_entry_point_id():
    global comfyd_process
    if 'comfyd_process' in globals() and comfyd_process:
        return gen_entry_point_id(comfyd_process.pid)
    else:
        return None
