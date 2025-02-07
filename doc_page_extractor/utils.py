import os

def ensure_dir(path: str) -> str:
  path = os.path.abspath(path)
  os.makedirs(path, exist_ok=True)
  return path