import os
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
import streamlit as st


@st.cache_resource
def load_internvl2_pipeline(model_dir: str, device: str = "CPU"):
    """Loads and caches the OpenVINO GenAI VLMPipeline."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory does not exist at: {model_dir}"
        )

    pipeline = ov_genai.VLMPipeline(model_dir, device)
    return pipeline


def extract_visual_features(
    pipeline, image_file, user_prompt: str = ""
) -> str:
    """Analyzes an input image along with user text using OpenVINO GenAI VLMPipeline."""
    if isinstance(image_file, (str, os.PathLike)):
        pil_img = Image.open(image_file).convert("RGB")
    elif hasattr(image_file, "read"):
        pil_img = Image.open(image_file).convert("RGB")
        image_file.seek(0)
    elif isinstance(image_file, Image.Image):
        pil_img = image_file.convert("RGB")
    else:
        raise ValueError("Unsupported image input format.")

    img_np = np.array(pil_img)
    ov_image_tensor = ov.Tensor(img_np)

    structured_prompt = (
        f"<image>\nAnalyze this apparel image in technical detail for manufacturing. "
        f"Extract garment type, construction lines, collar/neck style, sleeve style, pockets, closures, brim shape, stitching, and fabric characteristics. "
        f"Context provided by user: {user_prompt}\n\n"
        f"ALSO output these CAD attributes as JSON:\n"
        f"- garment_type: (t-shirt | bucket_hat | shirt | trousers | jacket | hoodie | dress | other)\n"
        f"- silhouette_shape: (rectangular | dome | tapered | flared | boxy | fitted)\n"
        f"- has_sleeves: (true | false)\n"
        f"- sleeve_length: (short | long | sleeveless)\n"
        f"- collar_type: (crew | v_neck | collarless | hood | other)\n"
        f"- has_pockets: (true | false)\n"
        f"- pocket_location: (chest | side | none)\n"
        f"- has_brim: (true | false)\n"
        f"- hem_type: (straight | curved | cinched)\n"
        f"- topstitching: (true | false)\n"
        f"Output these as key-value pairs after your analysis."
    )
    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 16384
    generation_config.temperature = 0
    

    result = pipeline.generate(
        prompt=structured_prompt,
        image=ov_image_tensor,
        generation_config=generation_config,
    )

    if hasattr(result, "texts"):
        return result.texts[0].strip()
    return str(result).strip()