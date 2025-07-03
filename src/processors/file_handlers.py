#%%
import textract
from pathlib import Path
from utils.file_tools import read_text_joined


SUPPORT_EXTENSIONS = ["txt", "pdf", "doc", "docx", 'md', "odt"]


def extract_text(file_path: str | Path, ext=None, max_chars: int = -1):
    fp = Path(file_path)
    if ext is None:
        ext = fp.suffix.lower().replace(".", "")
    
    assert ext in SUPPORT_EXTENSIONS, f"File extension must be one of {SUPPORT_EXTENSIONS}"

    if ext in ["txt", "md"]:
        text = read_text_joined(fp)
    else:
        byte_str = textract.process(fp, extension=ext)
        text = byte_str.decode("utf-8")
    
    return text[:max_chars]



# %%
