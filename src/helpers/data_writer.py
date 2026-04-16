from pathlib import Path
from typing import Union


def write_to_file(
  file: Path,
  data: Union[str, bytes],
  *,
  encoding: str = "utf-8",
  overwrite: bool = True,
  create_dirs: bool = True,
) -> None:
  if create_dirs:
    file.parent.mkdir(parents=True, exist_ok=True)

  if not overwrite and file.exists():
    raise FileExistsError(f"File already exists: {file}")

  if isinstance(data, bytes):
    with open(file, "wb") as f:
      f.write(data)
  else:
    with open(file, "w", encoding=encoding) as f:
      f.write(data)
