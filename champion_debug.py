"""champion_debug.py - draws detected champion positions onto a copy of each frame so you can
visually sanity-check champion_detection output (especially useful for the complex-scene
cases where the model's JSON is technically valid but the positions/IDs are wrong).

Color coding:
    played_champion        -> yellow
    friendly_champion_N    -> green
    enemy_champion_N       -> red
    anything else          -> white (shouldn't happen, but don't silently drop it)
"""

import logging
import os
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

logger = logging.getLogger("bettergameplay.champion_debug")

CIRCLE_RADIUS_PX = 14
CIRCLE_WIDTH_PX = 3

PLAYED_COLOR = (255, 215, 0)     # yellow
FRIENDLY_COLOR = (40, 200, 40)   # green
ENEMY_COLOR = (220, 30, 30)      # red
UNKNOWN_COLOR = (255, 255, 255)  # white fallback


def _color_for_champion_id(champion_id: str) -> Tuple[int, int, int]:
    if champion_id == "played_champion":
        return PLAYED_COLOR
    if champion_id.startswith("friendly_champion"):
        return FRIENDLY_COLOR
    if champion_id.startswith("enemy_champion"):
        return ENEMY_COLOR
    logger.warning("champion_debug: unrecognized champion id '%s', drawing in white", champion_id)
    return UNKNOWN_COLOR


def draw_champion_debug_frame(
    frame_path: str,
    champions: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Draw a circle + label for each detected champion on a copy of frame_path and save it
    to output_path. Silently skips entries missing x/y (defensive -- shouldn't happen given
    the prompt's contract, but this is a debug aid, not a critical path, so don't raise)."""
    with Image.open(frame_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)

        for champ in champions:
            x, y = champ.get("x"), champ.get("y")
            champ_id = champ.get("id", "unknown")
            if x is None or y is None:
                logger.warning(
                    "champion_debug: skipping '%s' in %s -- missing x/y",
                    champ_id, os.path.basename(frame_path),
                )
                continue

            cx, cy = x * width, y * height
            color = _color_for_champion_id(champ_id)

            draw.ellipse(
                [cx - CIRCLE_RADIUS_PX, cy - CIRCLE_RADIUS_PX, cx + CIRCLE_RADIUS_PX, cy + CIRCLE_RADIUS_PX],
                outline=color,
                width=CIRCLE_WIDTH_PX,
            )
            label = f"{champ_id} ({champ.get('confidence', '?')})"
            label_pos = (cx + CIRCLE_RADIUS_PX + 4, cy - CIRCLE_RADIUS_PX)
            # simple 1px black outline behind the text so it stays readable over any background
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((label_pos[0] + dx, label_pos[1] + dy), label, fill=(0, 0, 0))
            draw.text(label_pos, label, fill=color)

        img.save(output_path)