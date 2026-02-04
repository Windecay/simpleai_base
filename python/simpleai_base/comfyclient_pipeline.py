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
    if class_type == 'WanVideoSampler' and max_val > 300:
        return False

    defined_steps = _get_defined_steps(inputs)
    if defined_steps is not None:
        if class_type == 'WanVideoSampler':
            return not (max_val > defined_steps + 2 and max_val > defined_steps * 1.25)
        return not (max_val > defined_steps * 1.5 + 10)

    if class_type == 'WanVideoSampler' and total_steps_known is not None:
        return not (max_val > total_steps_known + 2 and max_val > total_steps_known * 1.25)

    return True

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


def queue_prompt(user_did, prompt, user_cert, extra_data=None):
    p = {"prompt": prompt, "client_id": user_did, "user_cert": user_cert}
    if extra_data:
        p["extra_data"] = extra_data
    data = json.dumps(p).encode('utf-8')
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post("http://{}/prompt".format(server_address()), data=data)
            if response.status_code == 200:
                return json.loads(response.read())
            else:
                print(f"{utils.now_string()} Error: {response.status_code} {response.text}")
                return None
    except httpx.RequestError as e:
        print(f"{utils.now_string()} httpx.RequestError: {e}")
        return None


def get_image(filename, subfolder, folder_type):
    params = httpx.QueryParams({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    })
    with httpx.Client() as client:
        response = client.get(f"http://{server_address()}/view", params=params)
        return response.read()


def get_history(prompt_id):
    with httpx.Client() as client:
        response = client.get("http://{}/history/{}".format(server_address(), prompt_id))
        return json.loads(response.read())


def get_images(user_did, ws, prompt, callback=None, total_steps=None, user_cert=None, extra_data=None):
    def format_progress(val):
        if isinstance(val, float):
            val = round(val, 1)
            if val.is_integer():
                return int(val)
        return val

    result  = queue_prompt(user_did, prompt, user_cert, extra_data)
    if result is None or 'prompt_id' not in result:
        if result:
            print(f'{utils.now_string()} [ComfyClient] Error in inference prompt: {result.get("error")}, {result.get("node_errors")}, user_did={user_did}')
        else:
            print(f'{utils.now_string()} [ComfyClient] Error in inference prompt: Result is None (Request Failed), user_did={user_did}')
        return None
    prompt_id = result['prompt_id']
    steps_str = f', total_steps={total_steps}' if total_steps is not None else ''
    print('{} [ComfyClient] Request and get prompt_id:{}{}'.format(utils.now_string(), prompt_id, steps_str))
    output_images = {}
    current_node = ''
    current_type = ''
    preview_nodes = ['KSampler', 'KSamplerAdvanced', 'SamplerCustomAdvanced', 'TiledKSampler', 'UltimateSDUpscale', 'UltimateSDUpscaleNoUpscale', 'FramePackSampler', 'WanVideoSampler', 'LanPaint_KSampler', 'LanPaint_SamplerCustom', 'LanPaint_KSamplerAdvanced', 'LanPaint_SamplerCustomAdvanced']
    save_nodes = ['SaveImageWebsocket', 'SaveImageWebsocketLazy', 'SaveVideoWebsocket']
    total_steps_known = total_steps
    current_step = 0
    current_total_steps = None
    finished_steps = 0
    step_offset = 0
    last_step_value = 0
    last_max_val = 0
    is_vhs_active = extra_data.get('is_vhs', False) if extra_data else False
    last_valid_image = None
    node_pass_count = {}
    node_last_val = {}

    while True:
        model_management.throw_exception_if_processing_interrupted()
        try:
            out = ws.recv()
        except Exception as e:
            print(f'{utils.now_string()} [ComfyClient] The connect was exception, restart and try again: {e}')
            last_err = e
            for attempt, sleep_s in enumerate([1.0, 3.0, 6.0], start=1):
                try:
                    time.sleep(sleep_s)
                    ws = websocket.WebSocket()
                    ws.connect("ws://{}/ws?clientId={}".format(server_address(), user_did))
                    out = ws.recv()
                    last_err = None
                    break
                except Exception as e2:
                    last_err = e2
            if last_err is not None:
                raise websocket.WebSocketException(str(last_err))

        if isinstance(out, str):
            message = json.loads(out)
            if not utils.echo_off:
                print(f'{utils.now_string()} [ComfyClient] feedback_message={message}')
            current_type = message['type']

            if current_type == 'VHS_latentpreview':
                pass
            data = message['data']
            if 'prompt_id' in data and data['prompt_id'] == prompt_id and 'node' in data:
                if data['node'] is not None:
                    current_node = data['node']
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
                if 'node' in data and data['node'] is not None:
                     current_node = data['node'] # Update current node if provided
                     last_val = node_last_val.get(current_node, -1)
                     if value < last_val:
                         node_pass_count[current_node] = node_pass_count.get(current_node, 0) + 1
                     node_last_val[current_node] = value
                if current_node:
                     node_to_check = current_node
                     if node_to_check not in prompt:
                         parts = node_to_check.split('.')
                         for i in range(len(parts) - 1, -1, -1):
                             test_id = '.'.join(parts[i:])
                             if test_id in prompt:
                                 node_to_check = test_id
                                 break
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
                    if value < last_step_value:
                        step_offset += last_max_val

                    last_step_value = value
                    last_max_val = max_val

                    current_step = step_offset + value
                    if total_steps_known:
                         current_total_steps = total_steps_known
                    else:
                         current_total_steps = step_offset + max_val

                    if callback is not None:
                        display_step = format_progress(current_step)
                        display_total = format_progress(current_total_steps if current_total_steps else total_steps_known)

                        try:
                            if is_vhs_active:
                                callback(display_step, display_total, None)
                            else:
                                callback(display_step, display_total, last_valid_image)
                        except Exception as e:
                            print(f"{utils.now_string()} [ComfyClient] Error calling callback in progress: {e}")

        else:
            if not utils.echo_off:
                length = len(out)
                length = 16 if length > 16 else length
                print(f'{utils.now_string()} [ComfyClient] feedback_stream({len(out)})={out[:length]}...')
            if current_node:
                node_to_check = current_node
                if node_to_check not in prompt:
                    parts = node_to_check.split('.')
                    for i in range(len(parts) - 1, -1, -1):
                        test_id = '.'.join(parts[i:])
                        if test_id in prompt:
                            node_to_check = test_id
                            break

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
                    elif callback is not None:
                        is_vhs = current_type == 'VHS_latentpreview' or is_vhs_active
                        if is_vhs and not utils.echo_off:
                            print(f'{utils.now_string()} [ComfyClient] VHS Frame received: len={len(out)}, node={current_node}, step={current_step}/{current_total_steps}')

                        if prompt[node_to_check]['class_type'] in preview_nodes or is_vhs:
                            if total_steps_known and not is_vhs:

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
                                if media_type == 'image':
                                    image_data = out[8:]
                                    if len(image_data) > 24 and image_data[0:2] != b'\xff\xd8' and image_data[24:26] == b'\xff\xd8':
                                        image_data = image_data[24:]
                                    elif len(image_data) > 20 and image_data[0:2] != b'\xff\xd8' and image_data[20:22] == b'\xff\xd8':
                                        image_data = image_data[20:]
                                    last_valid_image = np.array(Image.open(BytesIO(image_data)))
                                
                                callback(format_progress(display_step), format_progress(display_total), last_valid_image)
                                if is_vhs:
                                    time.sleep(0.02)
                            except Exception as e:
                                print(f"{utils.now_string()} [ComfyClient] Error decoding preview image: {e}")

    output_images_type = ['_'.join(k.split('_')[-2:]) for k, v in output_images.items()]
    output_images = {k: np.array(Image.open(BytesIO(v[-1]))) if 'image' in k else v[-1] for k, v in output_images.items()}
    print(f'{utils.now_string()} [ComfyClient] The ComfyTask:{prompt_id} has finished, get {len(output_images)} result: {output_images_type}')
    return output_images


def upload_file(file_path):
    if not os.path.exists(file_path):
        return None

    file_ext = os.path.splitext(file_path)[1]
    with open(file_path, 'rb') as f:
        file_content = f.read()
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Consistent naming logic with images_upload: upload_file_{hash[:32]}.{ext}
        filename = f'upload_file_{file_hash[:32]}{file_ext}'

        f.seek(0)
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
            with BytesIO() as output:
                pil_image.save(output, format="PNG")
                output.seek(0)
                files = {'image': (f'upload_image_{images.get_image_hash(k)[:32]}.png', output)}
                data = {'overwrite': 'true', 'type': 'input'}
                response = httpx.post("http://{}/upload/image".format(server_address()), files=files, data=data)
            filename2 = response.json()["name"]
            images.set_image_filename(k, filename2)
            result.update({k: filename2})
            print(f'{utils.now_string()} [ComfyClient] The ComfyTask:upload_input_image, {k}: {result[k]}')
        else:
            result.update({k: filename})
    return result


def process_flow(user_did, flow_name, params, images, callback=None, total_steps=None, user_cert=None, extra_data=None):
    global ws, client_id

    if ws is None or user_did != client_id or ws.status != 101:
        if ws is not None:
            print(f'{utils.now_string()} [ComfyClient] websocket status: {ws.status}, timeout:{ws.timeout}s. ready to reset.')
            ws.close()
        try:
            ws = websocket.WebSocket()
            ws.connect("ws://{}/ws?clientId={}".format(server_address(), user_did))
            client_id = user_did
        except Exception as e:
            print(f'{utils.now_string()} [ComfyClient] The connect_to_server has failed, sleep and try again: {e}')
            time.sleep(8)
            try:
                ws = websocket.WebSocket()
                ws.connect("ws://{}/ws?clientId={}".format(server_address(), user_did))
                client_id = user_did
            except Exception as e:
                print(f'{utils.now_string()} [ComfyClient] The connect_to_server has failed, restart and try again: {e}')
                time.sleep(12)
                try:
                    ws = websocket.WebSocket()
                    ws.connect("ws://{}/ws?clientId={}".format(server_address(), user_did))
                    client_id = user_did
                except Exception as e:
                    raise


    images_map = images_upload(images)
    params.update_params(images_map)

    # upload video and audio files if they are local paths
    current_params = params.get_params()
    files_to_upload = {}
    for key in ['video', 'audio']:
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
    try:
        prompt_str = params.convert2comfy(flow_name)
        if not utils.echo_off:
            pass #print(f'{utils.now_string()} [ComfyClient] ComfyTask prompt: {prompt_str}')
        images = get_images(user_did, ws, prompt_str, callback=callback, total_steps=total_steps, user_cert=user_cert, extra_data=extra_data)
        # ws.close()
    except websocket.WebSocketException as e:
        print(f'{utils.now_string()} [ComfyClient] The connect has been closed, restart and try again: {e}')
        ws = None

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
    assert _should_count_progress_as_sampler_step("WanVideoSampler", {"steps": 4}, 20, 4) is False
    assert _should_count_progress_as_sampler_step("WanVideoSampler", {"steps": "4"}, 4, 4) is True
    assert _should_count_progress_as_sampler_step("KSampler", {"steps": 20}, 20, None) is True
    assert _should_count_progress_as_sampler_step("KSampler", {"steps": 4}, 20, None) is False
