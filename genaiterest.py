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
    "analog photography": {
        "name": "sai-analog film",
        "prompt": "analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    },
    "architecture": {
        "name": "misc-architectural",
        "prompt": "architectural style {prompt} . clean lines, geometric shapes, minimalist, modern, architectural drawing, highly detailed",
        "negative_prompt": "curved lines, ornate, baroque, abstract, grunge"
    },
    "art deco": {
        "name": "artstyle-art deco",
        "prompt": "art deco style {prompt} . geometric shapes, bold colors, luxurious, elegant, decorative, symmetrical, ornate, detailed",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, modernist, minimalist"
    },
    "automotive": {
        "name": "ads-automotive",
        "prompt": "automotive advertisement style {prompt} . sleek, dynamic, professional, commercial, vehicle-focused, high-resolution, highly detailed",
        "negative_prompt": "noisy, blurry, unattractive, sloppy, unprofessional"
    },
    "clay art": {
        "name": "sai-craft clay",
        "prompt": "play-doh style {prompt} . sculpture, clay art, centered composition, Claymation",
        "negative_prompt": "sloppy, messy, grainy, highly detailed, ultra textured, photo"
    },
    "fashion": {
        "name": "ads-fashion editorial",
        "prompt": "fashion editorial style {prompt} . high fashion, trendy, stylish, editorial, magazine style, professional, highly detailed",
        "negative_prompt": "outdated, blurry, noisy, unattractive, sloppy"
    },
    "food photography": {
        "name": "ads-gourmet food photography",
        "prompt": "gourmet food photo of {prompt} . soft natural lighting, macro details, vibrant colors, fresh ingredients, glistening textures, bokeh background, styled plating, wooden tabletop, garnished, tantalizing, editorial quality",
        "negative_prompt": "cartoon, anime, sketch, grayscale, dull, overexposed, cluttered, messy plate, deformed"
    },
    "interior design": {
        "name": "ads-real estate",
        "prompt": "real estate photography style {prompt} . professional, inviting, well-lit, high-resolution, property-focused, commercial, highly detailed",
        "negative_prompt": "dark, blurry, unappealing, noisy, unprofessional"
    },
    "kirigami": {
        "name": "papercraft-kirigami",
        "prompt": "kirigami representation of {prompt} . 3D, paper folding, paper cutting, Japanese, intricate, symmetrical, precision, clean lines",
        "negative_prompt": "painting, drawing, 2D, noisy, blurry, deformed"
    },
    "long exposure photography": {
        "name": "photo-long exposure",
        "prompt": "long exposure photo of {prompt} . Blurred motion, streaks of light, surreal, dreamy, ghosting effect, highly detailed",
        "negative_prompt": "static, noisy, deformed, shaky, abrupt, flat, low contrast"
    },
    "luxury": {
        "name": "ads-luxury",
        "prompt": "luxury product style {prompt} . elegant, sophisticated, high-end, luxurious, professional, highly detailed",
        "negative_prompt": "cheap, noisy, blurry, unattractive, amateurish"
    },
    "minimalism": {
        "name": "misc-minimalist",
        "prompt": "minimalist style {prompt} . simple, clean, uncluttered, modern, elegant",
        "negative_prompt": "ornate, complicated, highly detailed, cluttered, disordered, messy, noisy"
    },
    "neon noir": {
        "name": "photo-neon noir",
        "prompt": "neon noir {prompt} . cyberpunk, dark, rainy streets, neon signs, high contrast, low light, vibrant, highly detailed",
        "negative_prompt": "bright, sunny, daytime, low contrast, black and white, sketch, watercolor"
    },
    "origami": {
        "name": "sai-origami",
        "prompt": "origami style {prompt} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
        "negative_prompt": "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
    },
    "pixel art": {
        "name": "sai-pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"
    },
    "pop art": {
        "name": "artstyle-pop art",
        "prompt": "pop Art style {prompt} . bright colors, bold outlines, popular culture themes, ironic or kitsch",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, minimalist"
    },
    "sci-fi": {
        "name": "futuristic-sci-fi",
        "prompt": "sci-fi style {prompt} . futuristic, technological, alien worlds, space themes, advanced civilizations",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, historical, medieval"
    },
    "street art": {
        "name": "artstyle-graffiti",
        "prompt": "graffiti style {prompt} . street art, vibrant, urban, detailed, tag, mural",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"
    },
    "tilt shift photography": {
        "name": "photo-tilt-shift",
        "prompt": "tilt-shift photo of {prompt} . selective focus, miniature effect, blurred background, highly detailed, vibrant, perspective control",
        "negative_prompt": "blurry, noisy, deformed, flat, low contrast, unrealistic, oversaturated, underexposed"
    },
    "typography": {
        "name": "artstyle-typography",
        "prompt": "typographic art {prompt} . stylized, intricate, detailed, artistic, text-based",
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
                    "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. Responses must be safe for work."
                },
                {
                    "role": "user",
                    "content": "Provide a bullet list of {} {} photography subjects, 8 words max per line".format(num_prompts, interest)
                }
            ],
            "stream": False,
            "max_tokens": 512
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
                    p = p.lstrip('0123456789.-• ')
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
        prompts.task_done()


def get_imagen():

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    i = 0
    while True:
        sdxl_future = sdxl_futures.get()
        future = sdxl_future["future"]
        prompt = sdxl_future["prompt"]
        sdxl_futures.task_done()
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


def generate_gallery(interests, num_images=15):

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
    print("Done generating the gallery!")

st.set_page_config(layout="wide", page_title="GenAIterest")

st.write("## GenAIterest - Powered by OctoAI")

interests = st.multiselect("Select your interests", sorted(sdxl_styles.keys()))

if st.button('Submit'):
    generate_gallery(interests)
