from typing import Generator
from dataclasses import dataclass
from shapely.geometry import Polygon


Point = tuple[float, float]

@dataclass
class Rectangle:
  lt: Point
  rt: Point
  lb: Point
  rb: Point

def iter_points(rect: Rectangle) -> Generator[Point, None, None]:
  yield rect.lt
  yield rect.lb
  yield rect.rb
  yield rect.rt

def intersection_area(rect1: Rectangle, rect2: Rectangle) -> float:
  poly1 = Polygon(iter_points(rect1))
  poly2 = Polygon(iter_points(rect2))
  intersection = poly1.intersection(poly2)
  if intersection.is_empty:
    return 0.0
  return intersection.area