import os
import zipfile
from io import BytesIO
import hashlib
from xml.etree import ElementTree

DOCX_SIGNATURE = b'PK\x03\x04'
OUTPUT_DIR = "recovered_docs_from_fragment"
MAX_DOCX_SIZE = 800 * 1024  # 800 KB
MAX_DOCX_TOTAL = 20  # recover only last 20 real unique .docx

def get_sha1(data):
    return hashlib.sha1(data).hexdigest()

def extract_text_from_docx_bytes(data):
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            if "word/document.xml" not in zf.namelist():
                return None
            xml_data = zf.read("word/document.xml")
            tree = ElementTree.fromstring(xml_data)
            paragraphs = tree.findall(
                ".//w:t",
                namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            )
            full_text = " ".join([p.text for p in paragraphs if p.text])
            return full_text.strip()
    except:
        return None

def is_valid_docx(chunk):
    try:
        with zipfile.ZipFile(BytesIO(chunk)) as zf:
            names = zf.namelist()
            return "word/document.xml" in names or "EncryptedPackage" in names
    except:
        return False

def recover_docx_from_fragment(fragment_path, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    with open(fragment_path, "rb") as f:
        data = f.read()

    recovered = []
    seen_texts = set()
    offset = 0
    count = 0

    while offset < len(data) and count < MAX_DOCX_TOTAL:
        sig_index = data.find(DOCX_SIGNATURE, offset)
        if sig_index == -1:
            break

        chunk = data[sig_index:sig_index + MAX_DOCX_SIZE]

        if not is_valid_docx(chunk):
            offset = sig_index + 4
            continue

        text = extract_text_from_docx_bytes(chunk)
        if not text or text.strip() in seen_texts:
            offset = sig_index + 4
            continue

        out_path = os.path.join(output_dir, f"recovered_{count + 1}.docx")
        with open(out_path, "wb") as out:
            out.write(chunk)

        recovered.append(out_path)
        seen_texts.add(text.strip())
        count += 1
        offset = sig_index + len(chunk)

    return recovered
