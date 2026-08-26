import json
import websocket
import uuid
import httpx
import time
import struct
import os
import numpy as np
import ldm_patched.modules.model_management as model_management
from io import BytesIO
from PIL import Image
import hashlib
from . import utils

COMFYUI_INPUT_DIRECTORY = None
COMFYUI_WEBSOCKET_TIMEOUT = 10.0
COMFYUI_SERVER_READY_TIMEOUT = 300.0
COMFYUI_SERVER_READY_POLL_INTERVAL = 0.5
COMFYUI_SERVER_READY_REQUEST_TIMEOUT = httpx.Timeout(3.0, connect=1.0)
COMFYUI_PROMPT_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
COMFYUI_PROMPT_ACCEPTANCE_TIMEOUT = 45.0
COMFYUI_PROMPT_ACCEPTANCE_POLL_INTERVAL = 1.0
COMFYUI_PROMPT_SUBMIT_ATTEMPTS = 2
COMFYUI_HISTORY_RECOVERY_TIMEOUT = 5.0
COMFYUI_HISTORY_RECOVERY_POLL_INTERVAL = 0.25
COMFYUI_WEBSOCKET_RECONNECT_DELAYS = (1.0, 3.0, 6.0, 10.0)
COMFYUI_PROMPT_MISSING_TIMEOUT = 30.0

_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


class ComfyServerUnavailableError(RuntimeError):
    pass

PREVIEW_NODE_CLASS_TYPES = {
    'KSampler',
    'KSamplerAdvanced',
    'SamplerCustomAdvanced',
    'TiledKSampler',
    'UltimateSDUpscale',
    'UltimateSDUpscaleNoUpscale',
    'FramePackSampler',
    'WanVideoSampler',
    'SCAIL2ScheduledLongVideo',
    'SCAIL2ScheduledLongVideoWithSAM',
    'SimpAIWanAnimateLoop',
    'LanPaint_KSampler',
    'LanPaint_SamplerCustom',
    'LanPaint_KSamplerAdvanced',
    'LanPaint_SamplerCustomAdvanced',
}

MULTI_PASS_PREVIEW_NODE_CLASS_TYPES = {
    'KSampler',
    'KSamplerAdvanced',
    'SamplerCustomAdvanced',
    'WanVideoSampler',
    'SCAIL2ScheduledLongVideo',
    'SCAIL2ScheduledLongVideoWithSAM',
    'SimpAIWanAnimateLoop',
}

SAVE_NODE_CLASS_TYPES = {
    'SaveImageWebsocket',
    'SaveImageWebsocketLazy',
    'SaveVideoWebsocket',
}

def set_input_directory(input_dir):
    global COMFYUI_INPUT_DIRECTORY
    if input_dir:
        COMFYUI_INPUT_DIRECTORY = os.path.abspath(str(input_dir))
    else:
        COMFYUI_INPUT_DIRECTORY = None

def _input_file_is_available(filename, expected_size):
    if not COMFYUI_INPUT_DIRECTORY:
        return False
    try:
        input_dir = os.path.abspath(COMFYUI_INPUT_DIRECTORY)
        target_path = os.path.abspath(os.path.join(input_dir, filename))
        if os.path.commonpath((input_dir, target_path)) != input_dir:
            return False
        return os.path.isfile(target_path) and os.path.getsize(target_path) == expected_size
    except Exception:
        return False

def _hash_file(file_path):
    file_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def connect_websocket(user_did):
    new_ws = websocket.WebSocket()
    new_ws.settimeout(COMFYUI_WEBSOCKET_TIMEOUT)
    new_ws.connect("ws://{}/ws?clientId={}".format(server_address(), user_did))
    return new_ws


def _close_websocket(target):
    if target is None:
        return
    close = getattr(target, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def wait_for_server_ready(
    timeout_seconds=COMFYUI_SERVER_READY_TIMEOUT,
    poll_interval=COMFYUI_SERVER_READY_POLL_INTERVAL,
    process_alive_callback=None,
):
    endpoint = "http://{}/system_stats".format(server_address())
    started_at = time.monotonic()
    deadline = started_at + max(0.0, float(timeout_seconds))
    last_error = None
    next_log_at = 0.0

    with httpx.Client(timeout=COMFYUI_SERVER_READY_REQUEST_TIMEOUT) as client:
        while True:
            model_management.throw_exception_if_processing_interrupted()
            if process_alive_callback is not None:
                try:
                    process_alive = bool(process_alive_callback())
                except Exception as exc:
                    raise ComfyServerUnavailableError(
                        f"Unable to inspect the Comfy backend process before submission: {exc}"
                    ) from exc
                if not process_alive:
                    raise ComfyServerUnavailableError(
                        "The Comfy backend process exited before its HTTP service became ready."
                    )

            try:
                response = client.get(endpoint)
                response.raise_for_status()
                elapsed = time.monotonic() - started_at
                if elapsed >= COMFYUI_SERVER_READY_POLL_INTERVAL:
                    print(f'{utils.now_string()} [ComfyClient] Comfy server ready after {elapsed:.1f}s: {endpoint}')
                return True
            except httpx.HTTPError as exc:
                last_error = exc

            now = time.monotonic()
            elapsed = now - started_at
            remaining = deadline - now
            if remaining <= 0:
                raise ComfyServerUnavailableError(
                    f"Timed out after {elapsed:.1f}s waiting for the Comfy HTTP service before submission: {last_error}"
                ) from last_error
            if elapsed >= next_log_at:
                print(
                    f'{utils.now_string()} [ComfyClient] waiting for Comfy server readiness '
                    f'elapsed={elapsed:.1f}s endpoint={endpoint}: {last_error}'
                )
                next_log_at = elapsed + 15.0
            time.sleep(min(max(0.05, float(poll_interval)), remaining))


def _connect_websocket_with_retry(client_id):
    last_error = None
    for attempt, sleep_seconds in enumerate((0.0, 2.0, 5.0), start=1):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        try:
            return connect_websocket(client_id)
        except Exception as exc:
            last_error = exc
            print(f'{utils.now_string()} [ComfyClient] websocket connect attempt {attempt} failed: {exc}')
    raise websocket.WebSocketException(str(last_error))

def _int_like(val):
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if val.is_integer() else None
    if isinstance(val, str):
        s = val.strip()
        if s.isdigit():
            return int(s)
        try:
            parsed = float(s)
            if parsed.is_integer():
                return int(parsed)
        except ValueError:
            pass
    return None

def _get_defined_steps(inputs):
    if not isinstance(inputs, dict):
        return None
    for key in ("steps", "sampling_steps", "sampler_steps", "num_steps", "step_count"):
        if key in inputs:
            parsed = _int_like(inputs.get(key))
            if parsed is not None and parsed > 0:
                return parsed
    return None

def _should_count_progress_as_sampler_step(class_type, inputs, max_val, total_steps_known):
    max_i = _int_like(max_val)
    if max_i is None:
        return True

    if class_type == 'WanVideoSampler' and max_i > 300:
        return False

    defined_steps = _get_defined_steps(inputs)
    if defined_steps is not None:
        if class_type == 'WanVideoSampler':
            return not (max_i > defined_steps + 2 and max_i > defined_steps * 1.25)
        return not (max_i > defined_steps * 1.5 + 10)

    total_steps_i = _int_like(total_steps_known)
    if total_steps_i is not None and total_steps_i > 0:
        return not (max_i > total_steps_i + 2 and max_i > total_steps_i * 1.25)

    if max_i > 300:
        return False

    return True

def _should_use_dynamic_stage_total(class_type):
    return class_type in MULTI_PASS_PREVIEW_NODE_CLASS_TYPES

def _resolve_prompt_node_id(node_id, prompt):
    if node_id in prompt:
        return node_id
    if not isinstance(node_id, str):
        return node_id

    parts = node_id.split('.')
    for end in range(len(parts) - 1, 0, -1):
        candidate = '.'.join(parts[:end])
        if candidate in prompt:
            return candidate

    for start in range(1, len(parts)):
        candidate = '.'.join(parts[start:])
        if candidate in prompt:
            return candidate

    return node_id

def _normalize_display_progress(step, total, last_step, last_total):
    step_i = _int_like(step)
    total_i = _int_like(total)
    last_step_i = _int_like(last_step)
    last_total_i = _int_like(last_total)

    if (
        total_i is not None
        and last_total_i is not None
        and total_i < last_total_i
    ):
        return last_step, last_total

    if (
        total_i is None
        and last_total_i is not None
        and step_i is not None
        and last_step_i is not None
        and step_i <= last_step_i
    ):
        return last_step, last_total

    if (
        step_i is not None
        and last_step_i is not None
        and total_i is not None
        and last_total_i is not None
        and total_i == last_total_i
        and step_i < last_step_i
    ):
        return last_step, last_total

    return step, total

class ComfyInputImage:
    default_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    default_image_hash = hashlib.sha256(default_image.tobytes()).hexdigest()

    def __init__(self, key_list):
        if not isinstance(key_list, list):
            raise ValueError("key_list must be a list")
        self.map = {}
        for key in key_list:
            self.map[key] = self.default_image
            self.map[f'{key}|hash'] = self.default_image_hash

    def get(self, key):
        return self.map.get(key, None)
    def set_image(self, key, image):
        if isinstance(image, np.ndarray):
            self.map[key] = image
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            self.map[f'{key}|hash'] = image_hash
        else:
            raise ValueError("image must be a np.ndarray")

    def set_image_filename(self, key, filename):
        image_hash = self.map[f'{key}|hash']
        self.map[f'{image_hash}|file'] = filename

    def get_image_hash(self, key):
        return self.map[f'{key}|hash']

    def get_image_filename(self, key):
        image_hash = self.map[f'{key}|hash']
        file_key = f'{image_hash}|file'
        return self.map.get(file_key, None)

    def exists(self, key):
        return key in self.map

    def get_key_list(self):
        return [k for k in self.map.keys() if not k.endswith('|hash') and not k.endswith('|file')]

    def len(self):
        return len(self.get_key_list())

def upload_mask(mask):
    with BytesIO() as output:
        mask.save(output)
        output.seek(0)
        files = {'mask': ('mask.jpg', output)}
        data = {'overwrite': 'true', 'type': 'example_type'}
        response = httpx.post("http://{}/upload/mask".format(server_address()), files=files, data=data)
    return response.json()


def get_job(prompt_id, timeout=5.0):
    with httpx.Client(timeout=timeout) as client:
        response = client.get("http://{}/api/jobs/{}".format(server_address(), prompt_id))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def wait_for_prompt_acceptance(prompt_id, timeout_seconds=COMFYUI_PROMPT_ACCEPTANCE_TIMEOUT):
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        try:
            job = get_job(prompt_id)
        except httpx.HTTPError:
            job = None
        if isinstance(job, dict) and job.get("id") == prompt_id:
            print(f'{utils.now_string()} [ComfyClient] recovered accepted prompt_id={prompt_id}, status={job.get("status")}')
            return job
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(COMFYUI_PROMPT_ACCEPTANCE_POLL_INTERVAL, remaining))


def queue_prompt(user_did, prompt, user_cert, extra_data=None, process_alive_callback=None):
    prompt_id = str(uuid.uuid4())
    p = {"prompt": prompt, "client_id": user_did, "user_cert": user_cert, "prompt_id": prompt_id}
    if extra_data:
        p["extra_data"] = extra_data
    data = json.dumps(p).encode('utf-8')
    for attempt in range(1, COMFYUI_PROMPT_SUBMIT_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=COMFYUI_PROMPT_HTTP_TIMEOUT) as client:
                response = client.post("http://{}/prompt".format(server_address()), data=data)
            if response.status_code == 200:
                result = json.loads(response.read())
                if result.get("prompt_id") != prompt_id:
                    print(f"{utils.now_string()} Error: Comfy returned a different prompt_id: {result}")
                    return None
                return result
            print(f"{utils.now_string()} Error: {response.status_code} {response.text}")
            return None
        except httpx.RequestError as e:
            print(f"{utils.now_string()} httpx.RequestError submitting prompt_id={prompt_id}, attempt={attempt}: {e}")
            connection_failed = isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
            ambiguous = not connection_failed
            if ambiguous:
                job = wait_for_prompt_acceptance(prompt_id)
                if job is not None:
                    return {
                        "prompt_id": prompt_id,
                        "recovered": True,
                        "job_status": job.get("status"),
                    }
            if attempt < COMFYUI_PROMPT_SUBMIT_ATTEMPTS:
                if connection_failed:
                    wait_for_server_ready(process_alive_callback=process_alive_callback)
                else:
                    time.sleep(1.0)
    return None


def get_image(filename, subfolder, folder_type):
    params = httpx.QueryParams({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    })
    with httpx.Client(timeout=60.0) as client:
        response = client.get(f"http://{server_address()}/view", params=params)
        response.raise_for_status()
        return response.read()


def get_history(prompt_id):
    with httpx.Client(timeout=20.0) as client:
        response = client.get("http://{}/history/{}".format(server_address(), prompt_id))
        response.raise_for_status()
        return json.loads(response.read())


def get_history_item(prompt_id):
    try:
        history = get_history(prompt_id)
    except Exception as e:
        print(f'{utils.now_string()} [ComfyClient] history check failed prompt_id={prompt_id}: {e}')
        return None

    if not isinstance(history, dict):
        return None

    item = history.get(prompt_id)
    if isinstance(item, dict):
        return item

    if isinstance(history.get("outputs"), dict) or isinstance(history.get("status"), dict):
        return history

    return None


def prompt_finished_in_history(prompt_id, context):
    history_item = get_history_item(prompt_id)
    if history_item is None:
        return False

    status = history_item.get("status")
    if isinstance(status, dict):
        completed = status.get("completed")
        status_str = status.get("status_str")
        print(f'{utils.now_string()} [ComfyClient] prompt_id={prompt_id} found in history after {context}: status={status_str}, completed={completed}')
    else:
        print(f'{utils.now_string()} [ComfyClient] prompt_id={prompt_id} found in history after {context}')
    return True


def _history_item_is_terminal(history_item):
    if not isinstance(history_item, dict):
        return False
    status = history_item.get("status")
    if not isinstance(status, dict):
        return False
    if status.get("completed") is True:
        return True
    return str(status.get("status_str") or "").lower() in {"success", "error", "failed", "cancelled"}


def _encode_history_video_payload(raw, media_format):
    format_code = {"webm": 10, "mp4": 11}.get(str(media_format).lower())
    if format_code is None:
        return None
    return struct.pack(">II", 4, format_code) + raw


def _recover_history_output_data(prompt_id, prompt, history_item, node_ids=None):
    history_outputs = history_item.get("outputs") if isinstance(history_item, dict) else None
    if not isinstance(history_outputs, dict):
        return {}

    requested_nodes = set(node_ids) if node_ids is not None else None
    recovered = {}
    image_extensions = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"}
    video_extensions = {"mp4", "webm", "mov", "mkv", "avi"}
    for history_node_id, node_outputs in history_outputs.items():
        node_id = _resolve_prompt_node_id(str(history_node_id), prompt)
        if requested_nodes is not None and node_id not in requested_nodes:
            continue
        if not isinstance(node_outputs, dict):
            continue
        prompt_node = prompt.get(node_id, {})
        title = str(prompt_node.get("_meta", {}).get("title") or node_id)
        for output_name, items in node_outputs.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                if not isinstance(filename, str) or not filename:
                    continue
                extension = os.path.splitext(filename)[1].lower().lstrip(".")
                if output_name == "images" or extension in image_extensions:
                    media_type = "image"
                elif output_name in ("video", "videos", "gifs") or extension in video_extensions:
                    media_type = "video"
                else:
                    continue
                try:
                    raw = get_image(filename, item.get("subfolder", ""), item.get("type", "output"))
                except Exception as exc:
                    print(f'{utils.now_string()} [ComfyClient] failed to recover output prompt_id={prompt_id}, filename={filename}: {exc}')
                    continue
                media_format = extension or "unknown"
                if media_type == "video":
                    raw = _encode_history_video_payload(raw, media_format)
                    if raw is None:
                        print(f'{utils.now_string()} [ComfyClient] unsupported history video format prompt_id={prompt_id}, filename={filename}')
                        continue
                media_name = f'{title}_{media_type}_{media_format}'
                recovered.setdefault(media_name, []).append(raw)
    return recovered


def recover_history_outputs(
    prompt_id,
    prompt,
    node_ids=None,
    timeout_seconds=COMFYUI_HISTORY_RECOVERY_TIMEOUT,
    poll_interval=COMFYUI_HISTORY_RECOVERY_POLL_INTERVAL,
):
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        history_item = get_history_item(prompt_id)
        if history_item is not None:
            recovered = _recover_history_output_data(prompt_id, prompt, history_item, node_ids)
            if recovered:
                print(f'{utils.now_string()} [ComfyClient] recovered {len(recovered)} output groups from history prompt_id={prompt_id}')
                return recovered
            if _history_item_is_terminal(history_item):
                return {}

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {}
        time.sleep(min(max(0.01, float(poll_interval)), remaining))


def get_images(user_did, ws, prompt, callback=None, total_steps=None, user_cert=None, extra_data=None, prompt_accepted_callback=None, _socket_holder=None, process_alive_callback=None):
    if _socket_holder is not None:
        _socket_holder[0] = ws

    def format_progress(val):
        if isinstance(val, float):
            val = round(val, 1)
            if val.is_integer():
                return int(val)
        return val

    def get_effective_stage_info(node_id, max_val):
        if node_id not in prompt:
            return None
        inputs = prompt[node_id].get("inputs", {})
        start_at_step = _int_like(inputs.get("start_at_step"))
        end_at_step = _int_like(inputs.get("end_at_step"))

        total_steps_i = _int_like(total_steps_known)
        effective_steps = None
        if start_at_step is not None and end_at_step is not None and end_at_step > start_at_step:
            effective_steps = end_at_step - start_at_step
        else:
            defined_steps = _get_defined_steps(inputs)
            if defined_steps is not None and defined_steps > 0:
                effective_steps = defined_steps

        if effective_steps is None:
            parsed_max = _int_like(max_val)
            if parsed_max is not None and parsed_max > 0:
                effective_steps = parsed_max
            elif isinstance(max_val, (int, float)) and max_val > 0:
                effective_steps = int(max_val)

        if total_steps_i is not None and total_steps_i > 0:
            if effective_steps is None or effective_steps > total_steps_i:
                effective_steps = total_steps_i

        if effective_steps is None or effective_steps <= 0:
            return None

        return {
            "start_at_step": start_at_step,
            "end_at_step": end_at_step,
            "effective_steps": effective_steps,
        }

    result = queue_prompt(
        user_did,
        prompt,
        user_cert,
        extra_data,
        process_alive_callback=process_alive_callback,
    )
    if result is None or 'prompt_id' not in result:
        if result:
            print(f'{utils.now_string()} [ComfyClient] Error in inference prompt: {result.get("error")}, {result.get("node_errors")}, user_did={user_did}')
        else:
            print(f'{utils.now_string()} [ComfyClient] Error in inference prompt: Result is None (Request Failed), user_did={user_did}')
        return None
    prompt_id = result['prompt_id']
    steps_str = f', total_steps={total_steps}' if total_steps is not None else ''
    print('{} [ComfyClient] Request and get prompt_id:{}{}'.format(utils.now_string(), prompt_id, steps_str))
    if prompt_accepted_callback is not None:
        try:
            prompt_accepted_callback(prompt_id, result)
        except Exception as e:
            print(f"{utils.now_string()} [ComfyClient] Error calling prompt_accepted_callback: {e}")
    output_images = {}
    received_output_nodes = set()
    output_names_by_node = {}
    history_recovery_needed = ws is None
    current_node = ''
    current_type = ''
    preview_nodes = PREVIEW_NODE_CLASS_TYPES
    save_nodes = SAVE_NODE_CLASS_TYPES
    total_steps_known = total_steps
    current_step = 0
    current_total_steps = None
    last_display_step = None
    last_display_total = None
    finished_steps = 0
    is_vhs_active = extra_data.get('is_vhs', False) if extra_data else False
    last_valid_image = None
    node_pass_count = {}
    node_last_val = {}
    node_display_ids = {}
    sampler_stages = []
    sampler_stage_info = {}

    def emit_progress(step, total, image=None, context="progress"):
        nonlocal last_display_step, last_display_total
        if callback is None:
            return
        display_step = format_progress(step)
        display_total = format_progress(total)
        display_step, display_total = _normalize_display_progress(
            display_step,
            display_total,
            last_display_step,
            last_display_total,
        )
        last_display_step = display_step
        last_display_total = display_total
        try:
            callback(display_step, display_total, image)
        except Exception as e:
            print(f"{utils.now_string()} [ComfyClient] Error calling callback in {context}: {e}")

    def reconnect_websocket(reason):
        nonlocal history_recovery_needed
        history_recovery_needed = True
        print(f'{utils.now_string()} [ComfyClient] websocket read interrupted, reconnect and continue prompt_id={prompt_id}: {reason}')
        _close_websocket(ws)
        if _socket_holder is not None:
            _socket_holder[0] = None

        attempt = 0
        missing_since = None
        last_err = reason
        while True:
            model_management.throw_exception_if_processing_interrupted()
            sleep_s = COMFYUI_WEBSOCKET_RECONNECT_DELAYS[
                min(attempt, len(COMFYUI_WEBSOCKET_RECONNECT_DELAYS) - 1)
            ]
            time.sleep(sleep_s)
            attempt += 1
            try:
                new_ws = connect_websocket(user_did)
            except Exception as reconnect_error:
                last_err = reconnect_error
            else:
                if _socket_holder is not None:
                    _socket_holder[0] = new_ws
                print(f'{utils.now_string()} [ComfyClient] websocket reconnected prompt_id={prompt_id}, attempt={attempt}')
                return new_ws, False

            job_lookup_succeeded = False
            job = None
            try:
                job = get_job(prompt_id)
                job_lookup_succeeded = True
            except httpx.HTTPError as job_error:
                if attempt == 1 or attempt % 6 == 0:
                    print(f'{utils.now_string()} [ComfyClient] job status unavailable while websocket is disconnected prompt_id={prompt_id}: {job_error}')

            if isinstance(job, dict):
                missing_since = None
                status = str(job.get("status") or "").lower()
                if status in _TERMINAL_JOB_STATUSES:
                    print(f'{utils.now_string()} [ComfyClient] prompt reached terminal status without websocket prompt_id={prompt_id}, status={status}')
                    return None, True
            elif job_lookup_succeeded:
                now = time.monotonic()
                if missing_since is None:
                    missing_since = now
                elif now - missing_since >= COMFYUI_PROMPT_MISSING_TIMEOUT:
                    raise websocket.WebSocketException(
                        f"prompt_id={prompt_id} is no longer present after websocket disconnect: {last_err}"
                    )

            if attempt == len(COMFYUI_WEBSOCKET_RECONNECT_DELAYS) or attempt % 6 == 0:
                print(f'{utils.now_string()} [ComfyClient] websocket still unavailable; waiting for the same task prompt_id={prompt_id}, attempt={attempt}: {last_err}')

    prompt_finished = False
    while True:
        model_management.throw_exception_if_processing_interrupted()
        if ws is None:
            ws, prompt_finished = reconnect_websocket("websocket unavailable after prompt submission")
            if prompt_finished:
                break
        try:
            out = ws.recv()
        except websocket.WebSocketTimeoutException:
            if prompt_finished_in_history(prompt_id, "websocket timeout"):
                history_recovery_needed = True
                break
            continue
        except Exception as e:
            ws, prompt_finished = reconnect_websocket(str(e))
            if prompt_finished:
                break
            continue

        if isinstance(out, str):
            if out == "":
                ws, prompt_finished = reconnect_websocket("empty text frame")
                if prompt_finished:
                    break
                continue
            try:
                message = json.loads(out)
            except json.JSONDecodeError as e:
                print(f'{utils.now_string()} [ComfyClient] Skip non-json websocket text for prompt_id={prompt_id}: {repr(out[:200])}, error={e}')
                continue
            if not isinstance(message, dict):
                continue
            if not utils.echo_off:
                print(f'{utils.now_string()} [ComfyClient] feedback_message={message}')
            current_type = message.get('type')
            data = message.get('data')
            if not current_type or not isinstance(data, dict):
                continue

            if current_type == 'VHS_latentpreview':
                pass
            if 'prompt_id' in data and data['prompt_id'] == prompt_id and 'node' in data:
                if data['node'] is not None:
                    event_node = data['node']
                    if current_type == 'executing':
                        node_display_ids[event_node] = data.get('display_node') or event_node
                    current_node = node_display_ids.get(event_node, event_node)
                    if current_type == 'executing':
                        node_pass_count[current_node] = 0
                        node_last_val[current_node] = -1
                elif current_type == 'executing':
                    break

            if current_type == 'VHS_latentpreview' and 'id' in data:
                current_node = data['id']


            if current_type == 'progress':
                value = data["value"]
                max_val = data["max"]

                is_sampler_step = False
                stage_key = None
                stage_node_id = None
                if 'node' in data and data['node'] is not None:
                     event_node = data['node']
                     current_node = node_display_ids.get(event_node, event_node)
                     last_val = node_last_val.get(current_node, -1)
                     if value < last_val:
                         node_pass_count[current_node] = node_pass_count.get(current_node, 0) + 1
                     node_last_val[current_node] = value
                if current_node:
                     node_to_check = _resolve_prompt_node_id(current_node, prompt)
                     if node_to_check in prompt:
                        class_type = prompt[node_to_check]['class_type']
                        if class_type in preview_nodes:
                            is_sampler_step = True
                            current_pass = node_pass_count.get(current_node, 0)

                            if ('UltimateSDUpscale' in class_type) and current_pass == 0:
                                is_sampler_step = False

                            # Additional safety check for mismatched step counts
                            inputs = prompt[node_to_check].get('inputs', {})
                            if not _should_count_progress_as_sampler_step(class_type, inputs, max_val, total_steps_known):
                                is_sampler_step = False

                            if is_sampler_step:
                                stage_node_id = node_to_check
                                stage_key = (stage_node_id, current_pass)

                if is_sampler_step:
                    if stage_key not in sampler_stage_info:
                        stage_meta = get_effective_stage_info(stage_node_id, max_val)
                        if stage_meta is None:
                            stage_meta = {"start_at_step": None, "end_at_step": None, "effective_steps": _int_like(max_val) or 1}

                        offset = 0
                        for k in sampler_stages:
                            offset += sampler_stage_info[k]["effective_steps"]
                        sampler_stage_info[stage_key] = {
                            "offset": offset,
                            "start_at_step": stage_meta["start_at_step"],
                            "end_at_step": stage_meta["end_at_step"],
                            "effective_steps": stage_meta["effective_steps"],
                            "first_value": None,
                        }
                        sampler_stages.append(stage_key)

                    stage_info = sampler_stage_info[stage_key]
                    start_at_step = stage_info.get("start_at_step")
                    end_at_step = stage_info.get("end_at_step")
                    effective_steps = stage_info["effective_steps"]

                    value_i = value
                    max_i = max_val
                    parsed_value = _int_like(value)
                    parsed_max = _int_like(max_val)
                    if parsed_value is not None:
                        value_i = parsed_value
                    if parsed_max is not None:
                        max_i = parsed_max

                    if stage_info.get("first_value") is None and isinstance(value_i, int):
                        stage_info["first_value"] = value_i

                    within = None
                    if (
                        start_at_step is not None
                        and end_at_step is not None
                        and end_at_step > start_at_step
                        and isinstance(value_i, int)
                        and value_i >= start_at_step
                    ):
                        within = value_i - start_at_step + 1
                        if within <= 0:
                            within = None
                    if within is None and isinstance(value_i, int):
                        first_value = stage_info.get("first_value")
                        if isinstance(first_value, int):
                            within = (value_i - first_value) + 1
                        else:
                            within = value_i + 1
                    if within is None:
                        within = 1

                    if within < 1:
                        within = 1
                    if effective_steps and within > effective_steps:
                        within = effective_steps

                    current_step = stage_info["offset"] + within

                    if total_steps_known and not _should_use_dynamic_stage_total(class_type):
                        current_total_steps = total_steps_known
                    else:
                        total_eff = 0
                        for k in sampler_stages:
                            total_eff += sampler_stage_info[k]["effective_steps"]
                        current_total_steps = total_eff

                    if callback is not None:
                        emit_progress(current_step, current_total_steps if current_total_steps else total_steps_known, None, "progress")

        else:
            if not utils.echo_off:
                length = len(out)
                length = 16 if length > 16 else length
                print(f'{utils.now_string()} [ComfyClient] feedback_stream({len(out)})={out[:length]}...')
            if current_node:
                node_to_check = _resolve_prompt_node_id(current_node, prompt)

                if node_to_check in prompt:
                    (media_type, media_format) = get_media_info(out[:8])
                    if prompt[node_to_check]['class_type'] in save_nodes:
                        media_name = f'{prompt[node_to_check]["_meta"]["title"]}_{media_type}_{media_format}'
                        images_output = output_images.get(media_name, [])
                        if media_type=='video':
                            images_output.append(out)
                        else:
                            images_output.append(out[8:])
                        output_images[media_name] = images_output
                        received_output_nodes.add(node_to_check)
                        output_names_by_node.setdefault(node_to_check, set()).add(media_name)
                    elif callback is not None:
                        is_vhs = current_type == 'VHS_latentpreview' or is_vhs_active
                        if is_vhs and not utils.echo_off:
                            print(f'{utils.now_string()} [ComfyClient] VHS Frame received: len={len(out)}, node={current_node}, step={current_step}/{current_total_steps}')

                        class_type = prompt[node_to_check]['class_type']
                        if class_type in preview_nodes or is_vhs:
                            total_steps_i = _int_like(total_steps_known)
                            if is_vhs and total_steps_i is not None and total_steps_i > 0:
                                current_step_i = _int_like(current_step)
                                display_total = total_steps_i
                                if current_step_i is not None and current_step_i > 0:
                                    display_step = min(current_step_i, total_steps_i)
                                else:
                                    display_step = 1
                            elif total_steps_known and not is_vhs and not _should_use_dynamic_stage_total(class_type):

                                if current_step > 0 and current_total_steps:
                                        display_step = current_step
                                        display_total = current_total_steps
                                else:
                                        finished_steps += 1
                                        display_step = finished_steps
                                        display_total = total_steps_known
                            else:
                                    # Complex logic for VHS or unknown steps
                                if current_total_steps is None or current_step <= current_total_steps:
                                    if current_step > 0:
                                        display_step = current_step
                                        display_total = current_total_steps 
                                        if display_total is None or display_total == 0:
                                                display_total = total_steps_known

                                    else:
                                        if not is_vhs:
                                            finished_steps += 1
                                        display_step = finished_steps if finished_steps > 0 else 1
                                        display_total = total_steps_known if total_steps_known else (current_total_steps if current_total_steps else '?')
                                else:
                                        display_step = current_step
                                        display_total = current_total_steps

                            try:
                                if media_type != 'image':
                                    continue
                                image_data = out[8:]
                                if len(image_data) > 24 and image_data[0:2] != b'\xff\xd8' and image_data[24:26] == b'\xff\xd8':
                                    image_data = image_data[24:]
                                elif len(image_data) > 20 and image_data[0:2] != b'\xff\xd8' and image_data[20:22] == b'\xff\xd8':
                                    image_data = image_data[20:]
                                last_valid_image = np.array(Image.open(BytesIO(image_data)))
                                
                                emit_progress(display_step, display_total, last_valid_image, "preview image")
                                if is_vhs:
                                    time.sleep(0.02)
                            except Exception as e:
                                print(f"{utils.now_string()} [ComfyClient] Error decoding preview image: {e}")

    expected_output_nodes = {
        node_id
        for node_id, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type") in save_nodes
    }
    missing_output_nodes = expected_output_nodes - received_output_nodes
    recovery_nodes = expected_output_nodes if history_recovery_needed else missing_output_nodes
    if recovery_nodes:
        try:
            recovered_outputs = recover_history_outputs(prompt_id, prompt, recovery_nodes)
        except Exception as exc:
            print(f'{utils.now_string()} [ComfyClient] history output recovery failed prompt_id={prompt_id}: {exc}')
            recovered_outputs = {}
        if recovered_outputs:
            for node_id in recovery_nodes:
                prompt_node = prompt.get(node_id, {})
                title = str(prompt_node.get("_meta", {}).get("title") or node_id)
                if any(name.startswith(f"{title}_") for name in recovered_outputs):
                    for output_name in output_names_by_node.get(node_id, set()):
                        output_images.pop(output_name, None)
            output_images.update(recovered_outputs)

    decoded_outputs = {}
    for name, values in output_images.items():
        if not values:
            continue
        try:
            decoded_outputs[name] = np.array(Image.open(BytesIO(values[-1]))) if 'image' in name else values[-1]
        except Exception as exc:
            print(f'{utils.now_string()} [ComfyClient] failed to decode recovered output prompt_id={prompt_id}, name={name}: {exc}')
    output_images = decoded_outputs
    output_images_type = ['_'.join(k.split('_')[-2:]) for k in output_images]
    print(f'{utils.now_string()} [ComfyClient] The ComfyTask:{prompt_id} has finished, get {len(output_images)} result: {output_images_type}')
    return output_images


def upload_file(file_path):
    if not os.path.exists(file_path):
        return None

    file_ext = os.path.splitext(file_path)[1]
    file_size = os.path.getsize(file_path)
    file_hash = _hash_file(file_path)
    filename = f'upload_file_{file_hash[:32]}{file_ext}'

    if _input_file_is_available(filename, file_size):
        print(f'{utils.now_string()} [ComfyClient] Reuse existing input file: {filename}')
        return filename

    with open(file_path, 'rb') as f:
        files = {'image': (filename, f)}
        data = {'overwrite': 'true', 'type': 'input'}
        response = httpx.post("http://{}/upload/image".format(server_address()), files=files, data=data)

    if response.status_code == 200:
        return response.json()["name"]
    return None

def images_upload(images):
    result = {}
    if images is None or images.len() == 0:
        return result
    for k in images.get_key_list():
        filename = images.get_image_filename(k)
        if filename is None:
            np_image = images.get(k)
            pil_image = Image.fromarray(np_image)
            filename2 = f'upload_image_{images.get_image_hash(k)[:32]}.png'
            with BytesIO() as output:
                pil_image.save(output, format="PNG")
                output.seek(0)
                if _input_file_is_available(filename2, output.getbuffer().nbytes):
                    print(f'{utils.now_string()} [ComfyClient] Reuse existing input image: {filename2}')
                else:
                    files = {'image': (filename2, output)}
                    data = {'overwrite': 'true', 'type': 'input'}
                    response = httpx.post("http://{}/upload/image".format(server_address()), files=files, data=data)
                    filename2 = response.json()["name"]
            images.set_image_filename(k, filename2)
            result.update({k: filename2})
            print(f'{utils.now_string()} [ComfyClient] The ComfyTask:upload_input_image, {k}: {result[k]}')
        else:
            result.update({k: filename})
    return result


def process_flow(user_did, flow_name, params, images, callback=None, total_steps=None, user_cert=None, extra_data=None, prompt_accepted_callback=None, process_alive_callback=None):
    wait_for_server_ready(process_alive_callback=process_alive_callback)
    images_map = images_upload(images)
    params.update_params(images_map)

    # upload video and audio files if they are local paths
    current_params = params.get_params()
    files_to_upload = {}
    for key in ['video', 'audio', 'reference_video', 'mask_video']:
        if key in current_params and isinstance(current_params[key], str) and os.path.exists(current_params[key]):
            print(f'{utils.now_string()} [ComfyClient] Uploading {key}: {current_params[key]}')
            new_filename = upload_file(current_params[key])
            if new_filename:
                files_to_upload[key] = new_filename
                print(f'{utils.now_string()} [ComfyClient] Uploaded {key} as: {new_filename}')
    if files_to_upload:
        params.update_params(files_to_upload)

    print(f'{utils.now_string()} [ComfyClient] Ready ComfyTask to process: workflow={flow_name}')
    current_params = params.get_params()
    for k, v in sorted(current_params.items()):
        if str(v) == 'placeholder.safetensors':
            continue
        if k.endswith('_strength'):
            base_key = k[:-9]
            if base_key in current_params and str(current_params[base_key]) == 'placeholder.safetensors':
                continue
        print(f'    {k} = {v}')
    socket_holder = [None]
    job_client_id = str(uuid.uuid4())
    try:
        prompt_str = params.convert2comfy(flow_name)
        if not utils.echo_off:
            pass #print(f'{utils.now_string()} [ComfyClient] ComfyTask prompt: {prompt_str}')
        try:
            socket_holder[0] = _connect_websocket_with_retry(job_client_id)
        except websocket.WebSocketException as websocket_error:
            print(f'{utils.now_string()} [ComfyClient] websocket unavailable before prompt submission; continue with job polling: {websocket_error}')
        images = get_images(
            job_client_id,
            socket_holder[0],
            prompt_str,
            callback=callback,
            total_steps=total_steps,
            user_cert=user_cert,
            extra_data=extra_data,
            prompt_accepted_callback=prompt_accepted_callback,
            _socket_holder=socket_holder,
            process_alive_callback=process_alive_callback,
        )
    except websocket.WebSocketException as e:
        print(f'{utils.now_string()} [ComfyClient] The connect has been closed, restart and try again: {e}')
        images = None
    finally:
        _close_websocket(socket_holder[0])

    imgs = []
    if images:
        images_keys = sorted(images.keys(), reverse=True)
        imgs = [images[key] for key in images_keys]
    else:
        print(f'{utils.now_string()} [ComfyClient] The ComfyTask:{flow_name} has no output images.')
    return imgs


def interrupt():
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post("http://{}/interrupt".format(server_address()))
            return
    except httpx.RequestError as e:
        print(f"{utils.now_string()} httpx.RequestError: {e}")
        return


def free(all=False):
    p = {"unload_models": all == True, "free_memory": True}
    data = json.dumps(p).encode('utf-8')
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post("http://{}/free".format(server_address()), data=data)
            return
    except httpx.RequestError as e:
        print(f"{utils.now_string()} httpx.RequestError: {e}")
        return

def setvars(vars):
    if not vars or not isinstance(vars, dict) or len(vars) == 0:
        return
    p = vars
    data = json.dumps(p).encode('utf-8')
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post("http://{}/setvars".format(server_address()), data=data)
            return
    except httpx.RequestError as e:
        print(f"{utils.now_string()} httpx.RequestError: {e}")
        return

def get_media_info(out):
    if out is None or len(out) < 8:
        return "unknown", "unknown"
    # 定义事件类型常量
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    PREVIEW_VIDEO = 4
    
    # 定义格式类型常量
    JPEG_FORMAT = 1
    PNG_FORMAT = 2
    WEBP_FORMAT = 3
    WEBM_FORMAT = 10
    MP4_FORMAT = 11
    
    event_type = struct.unpack(">I", out[:4])[0]
    format_type = struct.unpack(">I", out[4:8])[0]
    # 根据事件类型确定媒体类型
    if event_type == PREVIEW_IMAGE:
        media_type = "image"
    elif event_type == UNENCODED_PREVIEW_IMAGE:
        media_type = "unencoded_image"
    elif event_type == PREVIEW_VIDEO:
        media_type = "video"
    else:
        media_type = "unknown"
    
    # 根据格式类型确定格式名称
    if format_type == JPEG_FORMAT:
        format_name = "jpeg"
    elif format_type == PNG_FORMAT:
        format_name = "png"
    elif format_type == WEBP_FORMAT:
        format_name = "webp"
    elif format_type == WEBM_FORMAT:
        format_name = "webm"
    elif format_type == MP4_FORMAT:
        format_name = "mp4"
    else:
        format_name = "unknown"
    
    return media_type, format_name

WORKFLOW_DIR = 'workflows'
COMFYUI_ENDPOINT_IP = '127.0.0.1'
COMFYUI_ENDPOINT_PORT = '8187'
server_address = lambda: f'{COMFYUI_ENDPOINT_IP}:{COMFYUI_ENDPOINT_PORT}'
client_id = str(uuid.uuid4())
ws = None

if __name__ == "__main__":
    assert _resolve_prompt_node_id("aio_inpaint.0.0.5", {"aio_inpaint": {}}) == "aio_inpaint"
    assert _resolve_prompt_node_id("prefix.42", {"42": {}}) == "42"
    assert "KSampler" in MULTI_PASS_PREVIEW_NODE_CLASS_TYPES
    assert "WanVideoSampler" in MULTI_PASS_PREVIEW_NODE_CLASS_TYPES
    assert "SCAIL2ScheduledLongVideo" in PREVIEW_NODE_CLASS_TYPES
    assert "SCAIL2ScheduledLongVideoWithSAM" in PREVIEW_NODE_CLASS_TYPES
    assert "SCAIL2ScheduledLongVideoWithSAM" in MULTI_PASS_PREVIEW_NODE_CLASS_TYPES
    assert "SimpAIWanAnimateLoop" in PREVIEW_NODE_CLASS_TYPES
    assert "SimpAIWanAnimateLoop" in MULTI_PASS_PREVIEW_NODE_CLASS_TYPES
    assert _normalize_display_progress(6, 6, 7, 12) == (7, 12)
    assert _normalize_display_progress(6, None, 7, 12) == (7, 12)
    assert _normalize_display_progress(8, 12, 7, 12) == (8, 12)
    assert _should_count_progress_as_sampler_step("WanVideoSampler", {"steps": 4}, 20, 4) is False
    assert _should_count_progress_as_sampler_step("WanVideoSampler", {"steps": "4"}, 4, 4) is True
    assert _should_count_progress_as_sampler_step("KSampler", {"steps": 20}, 20, None) is True
    assert _should_count_progress_as_sampler_step("KSampler", {"steps": 4}, 20, None) is False
    assert _should_count_progress_as_sampler_step("SamplerCustomAdvanced", {}, 10002, 20) is False
    assert _should_count_progress_as_sampler_step("SamplerCustomAdvanced", {}, 10002, None) is False
