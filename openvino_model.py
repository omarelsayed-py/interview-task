import json
import os
import re

import streamlit as st

from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline


# ---------------------------------------------------------------------
# MODEL CACHE
# ---------------------------------------------------------------------

@st.cache_resource
def load_local_openvino_pipeline(
    model_dir: str,
    target_device: str = "AUTO",
):

    if not os.path.exists(model_dir):

        raise FileNotFoundError(
            f"Qwen3 model directory does not exist: {model_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=target_device,
        local_files_only=True,
    )

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=16384,
        do_sample=False,
    )

    return text_pipeline, tokenizer


# ---------------------------------------------------------------------
# JSON EXTRACTION
# ---------------------------------------------------------------------

def extract_and_parse_json(
    raw_text: str,
) -> dict:

    if not raw_text or not raw_text.strip():

        raise ValueError(
            "Qwen returned an empty response."
        )

    cleaned = raw_text.strip()

    # Remove Qwen reasoning blocks.
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL,
    ).strip()

    # Remove markdown fences.
    cleaned = re.sub(
        r"```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"```",
        "",
        cleaned,
    ).strip()

    # Locate JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:

        raise ValueError(
            "Qwen response does not contain a JSON object."
        )

    candidate = cleaned[
        start : end + 1
    ]

    try:

        parsed = json.loads(candidate)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Qwen returned malformed JSON: {e}"
        ) from e

    if not isinstance(parsed, dict):

        raise ValueError(
            "Qwen JSON root must be an object."
        )

    return parsed


# ---------------------------------------------------------------------
# TECH PACK GENERATION
# ---------------------------------------------------------------------

def generate_tech_pack_json(
    model_dir: str,
    product_description: str,
    visual_analysis: str,
) -> tuple[dict, str]:

    pipe, tokenizer = (
        load_local_openvino_pipeline(
            model_dir=model_dir,
            target_device="AUTO",
        )
    )

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

Never assume that every product is a bucket hat.

The product-specific measurement points must be selected according
to the detected product type.

Examples:

Bucket hat:
- crown circumference
- crown height
- brim width

Shirt:
- chest
- body length
- shoulder width
- sleeve length

Trousers:
- waist
- hip
- front rise
- inseam
- thigh
- leg opening

Jacket:
- chest
- body length
- shoulder
- sleeve length
- hem
- etc.

If a feature cannot be reliably identified from the image,
put it under uncertainties or assumptions.

Do not convert uncertainty into a confirmed manufacturing fact.

For example:

BAD:
"Closure: brass snap button"

GOOD:
"Closure type is not clearly visible; provisional recommendation:
snap or button depending on construction review."

CAD REQUIREMENTS:

cad_vector.primitives must contain real JSON arrays.

CORRECT:
"points": [[0, 0], [5, 0], [5, 5], [0, 5]]

WRONG:
"points": "[(0,0), (5,0), (5,5), (0,5)]"

Never output Python tuples.

Never output coordinates as a quoted string.

Never use "(x,y)" syntax.

Each point must be:

[x, y]

where x and y are JSON numbers.

All coordinates must be inside canvas_bounds.

Topstitching must use:

"role": "topstitching"
"linestyle": "--"

CAD geometry must represent the detected product rather than a
generic rectangle.

The CAD does not need to be production-grade patternmaking.
It must be a meaningful declarative 2D technical representation
of the observed garment.

OUTPUT FORMAT:

Return ONLY one valid JSON object.

Do not output markdown.

Do not output explanations.

Do not output <think> blocks.

Do not output text before or after the JSON.

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
    "points_of_measure": [
      "string"
    ],
    "sizes": {
      "S": {},
      "M": {},
      "L": {}
    }
  },

  "construction_details": [
    "string"
  ],

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

IMPORTANT:

The Python validation layer, NOT you, determines whether the
tech pack is valid.

Do not claim:

"validated": true

Do not add a fabricated validation status.

The quality_control section only records warnings and missing
information.

Keep lists concise.

Do not repeat the same observation multiple times.
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
- measurements are provisional unless scale is known
- product-specific sizing
- meaningful product-specific CAD
- real JSON coordinate arrays
- no Python tuples
- no quoted coordinate strings
- no <think> blocks
- JSON only
"""


    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


    formatted_prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


    outputs = pipe(
        formatted_prompt,
        return_full_text=False,
    )


    raw_text = outputs[0]["generated_text"]


    parsed_json = extract_and_parse_json(
        raw_text
    )


    return parsed_json, raw_text