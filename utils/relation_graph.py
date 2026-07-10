"""
Scene relation graph construction and utilities (Section 5.3).

Implements the Set-of-Mark (SoM) visual prompting technique for VLM
(Qwen-VL / GPT-4V), plus the mapping from fine-grained relations
(Stack, Lean, Hang, etc.) to the coarse categories (Contact, Support)
used in optimisation.

Supported VLM backends (both via OpenAI-compatible API):
  - Qwen-VL (DashScope): qwen-vl-max, qwen-vl-plus, qwen2.5-vl-72b-instruct
  - GPT-4V (OpenAI):     gpt-4-vision-preview, gpt-4o
"""

import base64
import io
import json
import random
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Fine-grained → coarse mapping (Section 5.3, CAST paper)
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
    Map a list of VLM relation outputs to a CAST constraint graph.

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

        # Normalize relationship name (strip whitespace, title-case)
        rel_type = rel_type.strip()
        # Map shorthand
        coarse = FINE_GRAINED_TO_COARSE.get(rel_type, 'Contact')

        if coarse == 'Support':
            # Directed: j supports i → edge (j → i)
            support_edges.add((j, i))
        elif coarse == 'Contact':
            # Bidirectional → canonical order
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
# VLM Prompt (Listing 1 from CAST paper — shared by Qwen and GPT-4V)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in object recognition and spatial reasoning.

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
Return ONLY a JSON array of objects, no other text:
[
  {"pair": [obj1_num, obj2_num], "relationship": "Stack", "reason": "explanation"},
  ...
]"""

USER_PROMPT_TEMPLATE = """### Object Details ###
I have labeled a bright numeric ID at the center for each visual object in the image.
Please analyze all relationships between the numbered objects and output JSON objects following the specified format.
Ensure each relationship includes:
1. The correct relationship type
2. A clear reason for the relationship
List all relationships as a JSON array."""


def build_vlm_messages(image_b64: str) -> List[dict]:
    """Construct OpenAI-compatible messages for VLM (works for both Qwen and GPT-4V)."""
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


# ---------------------------------------------------------------------------
# VLM API caller (OpenAI-compatible — works for Qwen DashScope + GPT-4V)
# ---------------------------------------------------------------------------

def _parse_json_from_response(text: str) -> List[dict]:
    """
    Robustly extract a JSON array from VLM response text.

    Handles markdown code fences, trailing commas, and mixed text.
    """
    # Try to find JSON array between ```json ... ``` fences
    code_fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_fence_match:
        text = code_fence_match.group(1)

    # Try to find a bare JSON array
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: try the whole text
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return []


def call_vlm_api(messages: List[dict],
                 api_key: str,
                 base_url: str,
                 model: str,
                 max_tokens: int = 2048,
                 temperature: float = 0.3) -> Optional[List[dict]]:
    """
    Call a VLM API (Qwen or GPT-4V) via OpenAI-compatible interface.

    Args:
        messages:    OpenAI-format message list
        api_key:     API key
        base_url:    API base URL
        model:       model name string
        max_tokens:  max output tokens
        temperature: sampling temperature

    Returns:
        List of parsed relation dicts, or None on failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        warnings.warn(
            "[VLM] openai package not installed. "
            "Install with: pip install openai"
        )
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        relations = _parse_json_from_response(content)

        if not relations:
            warnings.warn(
                f"[VLM] Could not parse relations from response: {content[:200]}..."
            )
        return relations

    except Exception as e:
        warnings.warn(f"[VLM] API call failed: {e}")
        return None


def query_vlm_with_som(image: np.ndarray,
                       objects: List,
                       api_key: str,
                       base_url: str,
                       model: str,
                       ensemble_trials: int = 3,
                       max_tokens: int = 2048,
                       temperature: float = 0.3,
                       verbose: bool = True) -> Dict:
    """
    Full VLM pipeline: SoM rendering → API call(s) → ensemble → constraint graph.

    Args:
        image:           (H, W, 3) uint8 RGB
        objects:         list of ObjectInfo or dicts with 'id' and 'mask'
        api_key:         VLM API key
        base_url:        VLM API base URL
        model:           VLM model name
        ensemble_trials: number of independent trials for majority voting
        max_tokens:      max output tokens per call
        temperature:     sampling temperature
        verbose:         print progress

    Returns:
        Constraint graph dict: {'nodes', 'contact_edges', 'support_edges'}
    """
    num_objects = len(objects)

    if num_objects < 2:
        if verbose:
            print("[VLM] Fewer than 2 objects — skipping relation reasoning.")
        return {
            'nodes': list(range(num_objects)),
            'contact_edges': [],
            'support_edges': [],
        }

    # Convert objects to SoM-compatible format
    som_objects = []
    for obj in objects:
        if hasattr(obj, 'id') and hasattr(obj, 'mask'):
            som_objects.append({'id': obj.id, 'mask': obj.mask})
        elif isinstance(obj, dict):
            som_objects.append(obj)
        else:
            som_objects.append({'id': len(som_objects), 'mask': None})

    # Collect relations across multiple trials (ensemble)
    all_relations: List[dict] = []

    for trial in range(ensemble_trials):
        if verbose:
            print(f"  [VLM] Trial {trial + 1}/{ensemble_trials} "
                  f"(model={model}) ...")

        # Render SoM with randomized colors for diversity
        som_image = render_set_of_mark(image, som_objects, colorize=True)
        image_b64 = encode_image_b64(som_image)
        messages = build_vlm_messages(image_b64)

        relations = call_vlm_api(
            messages=messages,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if relations:
            if verbose:
                print(f"    → {len(relations)} relations found")
            all_relations.extend(relations)
        else:
            if verbose:
                print(f"    → API call failed or no relations returned")

    if not all_relations:
        warnings.warn("[VLM] No relations obtained across all trials.")
        return {
            'nodes': list(range(num_objects)),
            'contact_edges': [],
            'support_edges': [],
        }

    # ---- Majority voting across trials (CAST paper ensemble strategy) ----
    # Count each unique relation (by pair + relationship type)
    relation_counts: Dict[Tuple, Dict[str, int]] = {}

    for r in all_relations:
        pair = tuple(sorted(r.get('pair', [])))
        rel_type = r.get('relationship', 'Stack').strip()
        if len(pair) != 2:
            continue
        if pair not in relation_counts:
            relation_counts[pair] = {}
        relation_counts[pair][rel_type] = relation_counts[pair].get(rel_type, 0) + 1

    # Keep relations that appear in ≥ half of the trials
    threshold = max(1, ensemble_trials // 2)
    confirmed_relations = []

    for pair, type_counts in relation_counts.items():
        total_votes = sum(type_counts.values())
        if total_votes >= threshold:
            # Pick the most common relationship type for this pair
            best_type = max(type_counts, key=type_counts.get)
            confirmed_relations.append({
                'pair': list(pair),
                'relationship': best_type,
                'reason': f'Majority vote: {total_votes}/{ensemble_trials} trials',
            })

    if verbose:
        print(f"  [VLM] Ensemble result: {len(confirmed_relations)} confirmed "
              f"relations (from {len(all_relations)} total across "
              f"{ensemble_trials} trials)")

    # Map to constraint graph
    return map_relations_to_graph(confirmed_relations, num_objects)
