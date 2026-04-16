from pathlib import Path


def write_to_file(
    file: Path,
    data: str | bytes,
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
            _ = f.write(data)
    else:
        with open(file, "w", encoding=encoding) as f:
            _ = f.write(data)
