import json
import os
import re
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline as hf_pipeline


# ============================================================
# MODEL LOADING (with device flexibility)
# ============================================================

def load_vlm_pipeline(model_dir: str, device: str = "CPU"):
    """Load InternVL2 VLM pipeline with automatic device fallback."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    try:
        vlm_pipe = ov_genai.VLMPipeline(model_dir, device)
        return vlm_pipe
    except Exception:
        # Fallback to CPU if device fails
        vlm_pipe = ov_genai.VLMPipeline(model_dir, "CPU")
        return vlm_pipe


def load_llm_pipeline(model_dir: str, device: str = "AUTO"):
    """Load Qwen3 LLM pipeline with automatic device fallback."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    
    try:
        model = OVModelForCausalLM.from_pretrained(
            model_dir,
            device=device,
            local_files_only=True,
            trust_remote_code=True,
            export=False,
        )
    except Exception:
        # Fallback to CPU
        model = OVModelForCausalLM.from_pretrained(
            model_dir,
            device="CPU",
            local_files_only=True,
            trust_remote_code=True,
            export=False,
        )
    
    text_pipeline = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=16384,
        do_sample=False,
    )
    
    return text_pipeline, tokenizer


# ============================================================
# JSON EXTRACTION (enhanced)
# ============================================================

def extract_and_parse_json(raw_text: str) -> dict:
    """Robustly extracts and parses JSON from model responses."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Model returned an empty response.")
    
    cleaned = raw_text.strip()
    
    # Remove think blocks
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL).strip()
    
    # Remove markdown fences
    cleaned = re.sub(r'```json\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'```', '', cleaned).strip()
    
    # Try direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Find first { and matching last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Response does not contain a valid JSON object.")
    
    candidate = cleaned[start:end + 1]
    
    try:
        parsed = json.loads(candidate)
        return parsed
    except json.JSONDecodeError as e:
        # Try fixing common issues
        try:
            # Remove trailing commas
            fixed = re.sub(r',\s*}', '}', candidate)
            fixed = re.sub(r',\s*]', ']', fixed)
            parsed = json.loads(fixed)
            return parsed
        except json.JSONDecodeError:
            raise ValueError(f"Model returned malformed JSON: {e}")


# ============================================================
# STAGE 1: VISUAL ANALYSIS (InternVL2)
# ============================================================

def extract_visual_features(pipeline, image_file, user_prompt: str = "") -> str:
    """Analyze an input image using OpenVINO GenAI VLMPipeline."""
    
    # Handle different image input types
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
        f"Extract garment type, construction lines, collar/neck style, sleeve style, "
        f"pockets, closures, brim shape, stitching, and fabric characteristics. "
        f"Context provided by user: {user_prompt}\n\n"
        f"ALSO output these CAD attributes:\n"
        f"garment_type: (t-shirt | bucket_hat | shirt | trousers | jacket | hoodie | dress | other)\n"
        f"silhouette_shape: (rectangular | dome | tapered | flared | boxy | fitted)\n"
        f"has_sleeves: (true | false)\n"
        f"sleeve_length: (short | long | sleeveless)\n"
        f"collar_type: (crew | v_neck | collarless | hood | other)\n"
        f"has_pockets: (true | false)\n"
        f"pocket_location: (chest | side | none)\n"
        f"has_brim: (true | false)\n"
        f"hem_type: (straight | curved | cinched)\n"
        f"topstitching: (true | false)"
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


# ============================================================
# STAGE 2: TECH PACK GENERATION (Qwen3)
# ============================================================

def generate_tech_pack_json(
    model_dir: str,
    product_description: str,
    visual_analysis: str,
) -> tuple[dict, str]:
    """Generate structured tech pack using Qwen3."""
    
    pipe, tokenizer = load_llm_pipeline(model_dir)
    
    system_prompt = r"""
You are an expert apparel manufacturing engineer.

Your task is to convert:
1. visual observations from an image
2. the buyer's product description

into a structured manufacturing tech pack.

IMPORTANT SOURCE PRIORITY:
1. Visual observations are the primary source for visible appearance.
2. User description provides design intent and requirements.
3. General apparel manufacturing knowledge may be used only for
   reasonable provisional recommendations.

Never invent physical measurements from pixels.
Never claim that an image provides an exact physical dimension unless
a known scale/reference is explicitly provided.
Measurements must be marked "provisional" when inferred without
physical measurement data.
Never confuse colorways with garment geometry.

The product-specific measurement points must be selected according
to the detected product type.

Examples:
Bucket hat: crown circumference, crown height, brim width
T-shirt: chest, body length, shoulder width, sleeve length
Trousers: waist, hip, front rise, inseam, thigh, leg opening
Jacket: chest, body length, shoulder, sleeve length, hem

CAD REQUIREMENTS:
cad_vector.primitives must contain real JSON arrays.
CORRECT: "points": [[0, 0], [5, 0], [5, 5], [0, 5]]
WRONG: "points": "[(0,0), (5,0), (5,5), (0,5)]"

Never output Python tuples or quoted coordinate strings.
Each point must be [x, y] with JSON numbers.
All coordinates inside canvas_bounds [-10, 10, -10, 10].
Topstitching: "role": "topstitching", "linestyle": "--"

OUTPUT FORMAT:
Return ONLY one valid JSON object.
No markdown, no explanations, no <think> blocks.

Required schema:
{
  "product_name": "string",
  "product_description": {
    "category": "apparel | headwear | accessory | footwear | unknown",
    "product_type": "string",
    "intended_use": "string",
    "design_summary": "string"
  },
  "design_evidence": {
    "observed_features": [],
    "inferred_features": [],
    "assumptions": [],
    "uncertainties": [],
    "conflicts_with_user_description": []
  },
  "cad_vector": {
    "canvas_bounds": [-10, 10, -10, 10],
    "primitives": [
      {
        "type": "polygon | line | bezier",
        "role": "garment_outline | panel | brim | pocket | collar | cuff | trim | seam | topstitching | construction_line",
        "points": [[0, 0], [5, 0], [5, 5]],
        "fill": "none",
        "stroke": "#000000",
        "linewidth": 1.2,
        "linestyle": "-"
      }
    ]
  },
  "materials_bom": [
    {
      "item": "string",
      "description": "string",
      "material_status": "observed | inferred | provisional",
      "color": "string"
    }
  ],
  "measurements_cm": {
    "size_system": "S/M/L",
    "measurement_status": "provisional",
    "points_of_measure": ["string"],
    "sizes": {"S": {}, "M": {}, "L": {}}
  },
  "construction_details": ["string"],
  "colorways": [
    {
      "name": "string",
      "main_color": "string",
      "secondary_color": "string",
      "construction_notes": []
    }
  ],
  "quality_control": {
    "warnings": [],
    "missing_information": []
  }
}
"""
    
    user_prompt = f"""
BUYER PRODUCT DESCRIPTION:
{product_description}

VISUAL ANALYSIS FROM INTERNVL2:
{visual_analysis}

Now generate the manufacturing tech pack.
Remember:
- visual evidence first
- user requirements second
- manufacturing knowledge third
- measurements provisional unless scale known
- product-specific sizing and CAD
- real JSON coordinate arrays
- no Python tuples or quoted coordinate strings
- JSON only
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    outputs = pipe(formatted_prompt, return_full_text=False)
    raw_text = outputs[0]["generated_text"]
    parsed_json = extract_and_parse_json(raw_text)
    
    return parsed_json, raw_text


# ============================================================
# SVG EXPORT (for vector output)
# ============================================================

def cad_vector_to_svg(cad_data: dict, output_filename: str = "garment_cad.svg") -> str:
    """Convert CAD vector to SVG file."""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    
    cad_vector = cad_data.get("cad_vector", cad_data)
    canvas_bounds = cad_vector.get("canvas_bounds", [-10, 10, -10, 10])
    primitives = cad_vector.get("primitives", [])
    
    xmin, xmax, ymin, ymax = canvas_bounds
    width, height = 800, 800
    padding = 40
    draw_width = width - (2 * padding)
    draw_height = height - (2 * padding)
    
    def map_x(x):
        return padding + ((x - xmin) / (xmax - xmin)) * draw_width
    
    def map_y(y):
        return height - (padding + ((y - ymin) / (ymax - ymin)) * draw_height)
    
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {width} {height}",
        "width": f"{width}px",
        "height": f"{height}px",
    })
    
    for prim in primitives:
        p_type = prim.get("type", "line")
        points = prim.get("points", [])
        stroke = prim.get("stroke", "#000000")
        linewidth = prim.get("linewidth", 1.5)
        linestyle = prim.get("linestyle", "-")
        fill = prim.get("fill", "none")
        dash = "5,4" if linestyle == "--" else "none"
        
        if p_type == "polygon" and len(points) >= 3:
            mapped = [(map_x(p[0]), map_y(p[1])) for p in points]
            d = f"M {mapped[0][0]:.1f},{mapped[0][1]:.1f} "
            d += " ".join([f"L {pt[0]:.1f},{pt[1]:.1f}" for pt in mapped[1:]])
            d += " Z"
            ET.SubElement(svg, "path", {
                "d": d, "stroke": stroke, "stroke-width": str(linewidth),
                "fill": fill, "stroke-dasharray": dash,
            })
        
        elif p_type in ["line", "bezier"] and len(points) >= 2:
            mapped = [(map_x(p[0]), map_y(p[1])) for p in points]
            d = f"M {mapped[0][0]:.1f},{mapped[0][1]:.1f} "
            d += " ".join([f"L {pt[0]:.1f},{pt[1]:.1f}" for pt in mapped[1:]])
            ET.SubElement(svg, "path", {
                "d": d, "stroke": stroke, "stroke-width": str(linewidth),
                "fill": "none", "stroke-dasharray": dash,
            })
    
    xml_str = ET.tostring(svg, encoding="utf-8")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    with open(output_filename, "w") as f:
        f.write(pretty)
    
    return output_filename