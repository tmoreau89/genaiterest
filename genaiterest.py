import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
import time
import threading
import queue
import random

# Let's import the OctoAI Python SDK
from octoai.client import Client

# Init OctoAI endpoint
client = Client()

# List of prompt entries to synthesize with SDXL
prompts = queue.Queue()

# SDXL futures
sdxl_futures = queue.Queue()

# SDXL styles
sdxl_styles = {
    "automotive": {
        "name": "ads-automotive",
        "prompt": "automotive advertisement style {prompt} . sleek, dynamic, professional, commercial, vehicle-focused, high-resolution, highly detailed",
        "negative_prompt": "noisy, blurry, unattractive, sloppy, unprofessional"
    },
    "interior design": {
        "name": "ads-real estate",
        "prompt": "real estate photography style {prompt} . professional, inviting, well-lit, high-resolution, property-focused, commercial, highly detailed",
        "negative_prompt": "dark, blurry, unappealing, noisy, unprofessional"
    },
    "architecture": {
        "name": "misc-architectural",
        "prompt": "architectural style {prompt} . clean lines, geometric shapes, minimalist, modern, architectural drawing, highly detailed",
        "negative_prompt": "curved lines, ornate, baroque, abstract, grunge"
    },
    "food photography": {
        "name": "ads-gourmet food photography",
        "prompt": "gourmet food photo of {prompt} . soft natural lighting, macro details, vibrant colors, fresh ingredients, glistening textures, bokeh background, styled plating, wooden tabletop, garnished, tantalizing, editorial quality",
        "negative_prompt": "cartoon, anime, sketch, grayscale, dull, overexposed, cluttered, messy plate, deformed"
    },
    "fashion": {
        "name": "ads-fashion editorial",
        "prompt": "fashion editorial style {prompt} . high fashion, trendy, stylish, editorial, magazine style, professional, highly detailed",
        "negative_prompt": "outdated, blurry, noisy, unattractive, sloppy"
    },
    "minimalism": {
        "name": "misc-minimalist",
        "prompt": "minimalist style {prompt} . simple, clean, uncluttered, modern, elegant",
        "negative_prompt": "ornate, complicated, highly detailed, cluttered, disordered, messy, noisy"
    },
    "tilt shift photography": {
        "name": "photo-tilt-shift",
        "prompt": "tilt-shift photo of {prompt} . selective focus, miniature effect, blurred background, highly detailed, vibrant, perspective control",
        "negative_prompt": "blurry, noisy, deformed, flat, low contrast, unrealistic, oversaturated, underexposed"
    },
    "graffiti": {
        "name": "artstyle-graffiti",
        "prompt": "graffiti style {prompt} . street art, vibrant, urban, detailed, tag, mural",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"
    }
}


# A helper function that reads a PIL Image objects and returns a base 64 encoded string
def encode_image(image: Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64


# A helper function that reads a base64 encoded string and returns a PIL Image object
def decode_image(image_str: str) -> Image:
    return Image.open(BytesIO(b64decode(image_str)))


def get_prompts(interests, num_prompts):
    # Async call to Llama 2
    futures = []
    for interest in interests:
        # Ask LLAMA for n subject ideas
        llama_inputs = {
            "model": "llama-2-7b-chat",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                },
                {
                    "role": "user",
                    "content": "Provide a list of {} {} photography subjects, 7 words max per line, no numbering".format(num_prompts, interest)
                }
            ],
            "stream": False,
            "max_tokens": 300
        }
        # Send to Llama 2 endpoint
        future = client.infer_async(endpoint_url="https://ga-demo-llama27b-4jkxk521l3v1.octoai.run/v1/chat/completions", inputs=llama_inputs)
        futures.append((future, interest))

    # Read the inputs back
    while len(futures):
        for future, interest in futures:
            if client.is_future_ready(future):
                outputs = client.get_future_result(future)
                llama2_response = outputs.get('choices')[0].get("message").get('content')
                # Derive the prompt list
                prompt_list = llama2_response.split('\n')
                # Do some cleaning
                for p in prompt_list:
                    p = p.lstrip('0123456789.- ')
                    if p != "" and "ere are" not in p:
                        prompts.put({
                            "prompt": p,
                            "style": interest
                        })
                # Remove futures tuple
                futures.remove((future, interest))
        time.sleep(.1)


def launch_imagen():

    SDXL_payload = {
        "prompt": "A photo of an octopus playing chess",
        "negative_prompt":"Blurry photo, distortion, low-res, bad quality",
        "style_preset":"",
        "cfg_scale":7.5,
        "steps":20,
        "seed": random.randint(0, 1024)
    }

    while True:
        p = prompts.get()
        prompt = p["prompt"]
        style = p["style"]
        SDXL_payload["prompt"] = sdxl_styles[style]["prompt"].replace("{prompt}", prompt)
        SDXL_payload["negative_prompt"] = sdxl_styles[style]["negative_prompt"]
        future = client.infer_async(
            endpoint_url="https://ga-demo-sdxl-4jkxk521l3v1.octoai.run/predict",
            inputs=SDXL_payload
        )
        sdxl_futures.put({
            "future": future,
            "prompt": prompt
        })
        print("Launched SDXL to generate: {}".format(prompt))


def get_imagen():

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    i = 0
    while True:
        sdxl_future = sdxl_futures.get()
        future = sdxl_future["future"]
        prompt = sdxl_future["prompt"]
        if client.is_future_ready(future):
            result = client.get_future_result(future)
            cols[i%len(cols)].image(
                decode_image(result["completion"]["image_0"]),
                caption=prompt
            )
            i += 1
            print("Finished SDXL generation of: {}".format(prompt))
        else:
            sdxl_futures.put(sdxl_future)


def generate_gallery(interests, num_images=18):

    # Start the threads to do image gen async
    t1 = threading.Thread(target=launch_imagen)
    t2 = threading.Thread(target=get_imagen)
    st.runtime.scriptrunner.add_script_run_ctx(t1)
    st.runtime.scriptrunner.add_script_run_ctx(t2)
    t1.start()
    t2.start()

    # Generate the prompts
    get_prompts(interests, num_images)

    # Wait for it all to finish
    prompts.join()
    sdxl_futures.join()
    print('All work completed')


st.set_page_config(layout="wide", page_title="GenAIterest")

st.write("## GenAIterest - Powered by OctoAI")

interests = st.multiselect("Select your interests", [
    "automotive",
    "interior design",
    "architecture",
    "food photography",
    "fashion",
    "minimalism",
    "tilt shift photography",
    "graffiti"
])

if st.button('Submit'):
    generate_gallery(interests)
