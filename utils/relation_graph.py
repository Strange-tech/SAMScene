"""
Scene relation graph construction and utilities (Section 5.3).

Implements the Set-of-Mark (SoM) visual prompting technique for GPT-4V,
plus the mapping from fine-grained relations (Stack, Lean, Hang, etc.)
to the coarse categories (Contact, Support) used in optimisation.
"""

import base64
import io
import json
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Fine-grained → coarse mapping
# ---------------------------------------------------------------------------

FINE_GRAINED_TO_COARSE = {
    'Stack':      'Support',
    'Lean':       'Support',
    'Hang':       'Support',
    'Clamped':    'Contact',
    'Contained':  'Contact',
    'Edge/Point': 'Contact',
}


def map_relations_to_graph(relations: List[dict],
                           num_objects: int) -> Dict:
    """
    Map a list of GPT-4V relation outputs to a CAST constraint graph.

    Args:
        relations: list of dicts with keys 'pair', 'relationship', 'reason'
                   e.g. [{'pair': [1, 2], 'relationship': 'Stack', 'reason': '...'}]
        num_objects: total number of objects N

    Returns:
        {'nodes': list(range(N)), 'contact_edges': [...], 'support_edges': [...]}
    """
    contact_edges = set()
    support_edges = set()

    for r in relations:
        pair = r.get('pair', [])
        rel_type = r.get('relationship', 'Stack')
        if len(pair) != 2:
            continue
        i, j = pair[0], pair[1]
        coarse = FINE_GRAINED_TO_COARSE.get(rel_type, 'Contact')

        if coarse == 'Support':
            # Directed: j supports i → edge (j → i)
            support_edges.add((j, i))
        elif coarse == 'Contact':
            # Bidirectional
            contact_edges.add((i, j) if i < j else (j, i))

    return {
        'nodes': list(range(num_objects)),
        'contact_edges': sorted(contact_edges),
        'support_edges': sorted(support_edges),
    }


# ---------------------------------------------------------------------------
# Set-of-Mark rendering
# ---------------------------------------------------------------------------

def render_set_of_mark(image: np.ndarray,
                       objects: List,
                       colorize: bool = True) -> np.ndarray:
    """
    Render numbered masks overlaying the image (SoM prompting, Yang et al. 2023).

    Args:
        image:   (H, W, 3) uint8 RGB
        objects: list with each having {'mask': (H,W) binary, 'id': int}
        colorize: if True, use random colors for masks; else white outlines

    Returns:
        (H, W, 3) uint8 RGB with numbered overlays
    """
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image).convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for obj in objects:
        mask = obj.get('mask', None)
        obj_id = obj.get('id', 0)

        if colorize:
            r, g, b = random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)
        else:
            r, g, b = 255, 255, 255

        if mask is not None and mask.any():
            # Draw mask outline
            mask_img = Image.fromarray((mask * 128).astype(np.uint8), mode='L')
            colored = Image.new('RGBA', (w, h), (r, g, b, 80))
            overlay = Image.composite(colored, overlay, mask_img)

            # Find mask centroid for label placement
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                bbox = draw.textbbox((cx, cy), str(obj_id), font=font)
                # Draw background circle
                r_rect = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 2 + 4
                draw.ellipse([cx - r_rect, cy - r_rect,
                              cx + r_rect, cy + r_rect],
                             fill=(r, g, b, 200))
                draw.text((cx - 5, cy - 7), str(obj_id),
                          fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(pil_img, overlay).convert('RGB')
    return np.array(result)


# ---------------------------------------------------------------------------
# GPT-4V prompt (Listing 1 from the paper)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert in object recognition and spatial reasoning.

### Task Description ###
Analyze an image with numbered objects and determine their relationships.
For each pair of related objects, output a JSON object containing the relationship details.
Ensure you output all possible relationships, even those that may be difficult to judge or less obvious.

### Relationship Definitions ###
1. **Stack**: Object 1 is on top of Object 2 (Object 2 supports Object 1 from below)
2. **Lean**: Object 1 is leaning against Object 2 (Object 2 supports Object 1 laterally)
3. **Hang**: Object 1 is hanging from Object 2 (Object 2 supports Object 1 from above)
4. **Clamped**: Object 1 is clamped by Object 2 (Object 2 grips Object 1 on multiple sides)
5. **Contained**: Object 1 is inside Object 2 (Object 2 encloses Object 1)
6. **Edge/Point**: Object 1 is touching Object 2 at an edge or point (minimal contact, no significant support)

### Important Note ###
Only objects that are in contact with each other should have a relationship.
For each relationship, always ensure:
1. Use the correct relationship type.
2. Provide a clear explanation of the relationship.
3. For cases that are in contact but hard to choose which type, use "Stack".

### Output Format ###
Return a JSON array of objects:
[
  {"pair": [obj1_num, obj2_num], "relationship": "Stack", "reason": "explanation"},
  ...
]
"""

USER_PROMPT_TEMPLATE = """
### Object Details ###
I have labeled a bright numeric ID at the center for each visual object in the image.
Please analyze all relationships between the numbered objects and output JSON objects following the specified format.
Ensure each relationship includes:
1. The correct relationship type
2. A clear reason for the relationship
List all relationships as a JSON array.
"""


def build_gpt4v_messages(image_b64: str) -> List[dict]:
    """Construct OpenAI-format messages for GPT-4V."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT_TEMPLATE},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        },
    ]


def encode_image_b64(image: np.ndarray) -> str:
    """Encode an RGB image array to a base64 PNG string."""
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')
