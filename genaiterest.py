import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
import time

# Let's import the OctoAI Python SDK
from octoai.client import Client

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

def generate_gallery(interests):

    # Init OctoAI endpoint
    client = Client()

    # Gallery columns
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    SDXL_payload = {
        "prompt": "A photo of an octopus playing chess",
        "negative_prompt":"Blurry photo, distortion, low-res, bad quality",
        "style_preset":"",
        "cfg_scale":7.5,
        "steps":20
    }

    # For number of images
    num_images = 10

    # Async call
    futures = []
    for i in range(0, len(interests)):
        # Derive Category
        category = interests[i]
        # Ask LLAMA for n subject ideas
        llama_inputs = {
            "model": "llama-2-13b-chat",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
                {
                    "role": "user",
                    "content": "Provide a consise list of {} {} photography subjects, 12 words per item at most".format(num_images, category)
                }
            ],
            "stream": False,
            "max_tokens": 512
        }
        # Send to LLAMA endpoint and do some post processing on the response stream
        outputs = client.infer(endpoint_url="https://ga-demo-llama2-4jkxk521l3v1.octoai.run/v1/chat/completions", inputs=llama_inputs)

        # Get the Llama 2 output
        categories = outputs.get('choices')[0].get("message").get('content')
        # Derive the prompt list (only 10 items)
        prompt_list = categories.split('\n')[2:12]
        # Remove the bullet point numbering
        prompt_list = [x.split('. ')[1] for x in prompt_list]
        for p in prompt_list:
            print(p)

        for j in range(0, num_images):
            SDXL_payload["prompt"] = sdxl_styles[category]["prompt"].replace("{prompt}", prompt_list[j])
            SDXL_payload["negative_prompt"] = sdxl_styles[category]["negative_prompt"]
            future = client.infer_async(
                endpoint_url="https://ga-demo-sdxl-4jkxk521l3v1.octoai.run/predict",
                inputs=SDXL_payload
            )
            futures.append(future)

        img_idx = 0
        while len(futures):
            for future in futures:
                if client.is_future_ready(future):
                    result = client.get_future_result(future)
                    cols[img_idx % len(cols)].image(decode_image(result["completion"]["image_0"]))
                    futures.remove(future)
                    img_idx += 1
            time.sleep(.1)

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
