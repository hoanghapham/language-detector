from pathlib import Path
import os
import sys
import json
import unicodedata


def list_files(input_path: Path | str, extensions) -> list[Path]:
    """Return a sorted list of PosixPaths"""
    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    files = [
        file for file in sorted(list(input_path.glob("**/*"))) 
        if file.suffix in extensions
        and file.is_file()
    ]
        
    return files


def read_ndjson_file(input_path: Path | str) -> list[dict]:
    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    return data


def read_json_file(input_path: Path | str) -> dict:
    with open(input_path, "r") as f:
        data = json.load(f)
    return data


def write_ndjson_file(data: list[dict], output_path: Path | str):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def write_json_file(data: dict, output_path: Path | str):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def read_text_lines(input_file: Path | str) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def read_text_joined(input_file: Path | str) -> str:
    with open(input_file, "r", encoding="utf-8") as f:
        text = "".join(list(f.readlines()))
    return text


def write_text_file(text: str, output_path: Path | str) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def write_list_to_text_file(lst: list[str], output_path: Path | str, linebreak=True) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
        
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lst:
            f.write(line)
            if linebreak:
                f.write("\n")
