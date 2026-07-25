from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .bounding_box_utils import normalized_to_pixel_xyxy


def draw_detection(
    image: Image.Image,
    box: list[float],
    label: str,
    confidence: float,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    x1, y1, x2, y2 = normalized_to_pixel_xyxy(box, width, height)

    line_width = max(2, round(min(width, height) / 80))
    draw.rectangle((x1, y1, x2, y2), outline="red", width=line_width)

    text = f"Digit {label} | {confidence:.1%}"
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
    text_height = bottom - top
    text_width = right - left
    label_y = max(0, y1 - text_height - 6)
    draw.rectangle(
        (x1, label_y, min(width - 1, x1 + text_width + 6), label_y + text_height + 6),
        fill="red",
    )
    draw.text((x1 + 3, label_y + 3), text, fill="white", font=font)
    return output
