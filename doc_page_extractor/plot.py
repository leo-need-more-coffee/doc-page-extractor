from typing import Iterable
from PIL import Image, ImageDraw
from .types import Layout

def plot(image: Image, layouts: Iterable[Layout]):
  draw = ImageDraw.Draw(image, mode="RGBA")
  for layout in layouts:
    x0, y0 = layout.origin
    w, h = layout.size
    rect = [(x0, y0), (x0 + w, y0 + h)]
    draw.rectangle(rect, outline=(255,0, 0, 128))