import os
import zipfile
import math
from collections import Counter
from io import BytesIO
from xml.etree import ElementTree
import olefile

def calculate_entropy(data):
    if not data:
        return 0
    counter = Counter(data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counter.values())

def extract_docx_text(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        with zipfile.ZipFile(BytesIO(data)) as zf:
            if "word/document.xml" not in zf.namelist():
                return "[Error]: No document.xml found"
            xml_data = zf.read("word/document.xml")
            tree = ElementTree.fromstring(xml_data)
            paragraphs = tree.findall(
                './/w:t',
                namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            )
            return " ".join([p.text for p in paragraphs if p.text])
    except Exception as e:
        return f"[Error]: {e}"

def extract_docx_images(path):
    images = []
    try:
        with open(path, 'rb') as f:
            data = f.read()
        with zipfile.ZipFile(BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.startswith("word/media/"):
                    images.append(os.path.basename(name))
    except Exception as e:
        images.append(f"[Error]: {e}")
    return images

def analyze_docx(path):
    result = {
        "type": "docx",
        "valid_zip": False,
        "has_document_xml": False,
        "encrypted": False,
        "extracted_text": "",
        "images": []
    }
    try:
        with open(path, 'rb') as f:
            data = f.read()
        with zipfile.ZipFile(BytesIO(data)) as zf:
            result["valid_zip"] = True
            names = zf.namelist()
            result["has_document_xml"] = "word/document.xml" in names
            result["encrypted"] = "EncryptedPackage" in names
            if result["has_document_xml"] and not result["encrypted"]:
                result["extracted_text"] = extract_docx_text(path)
            result["images"] = extract_docx_images(path)
    except Exception as e:
        result["error"] = str(e)
    return result

def analyze_doc(path):
    result = {
        "type": "doc",
        "valid_ole": False,
        "has_word_stream": False,
        "encrypted": False
    }
    try:
        if olefile.isOleFile(path):
            result["valid_ole"] = True
            ole = olefile.OleFileIO(path)
            streams = ole.listdir()
            result["has_word_stream"] = any('WordDocument' in '/'.join(s) for s in streams)
            if ole.exists('EncryptionInfo') or ole.exists('EncryptedPackage'):
                result["encrypted"] = True
            ole.close()
    except Exception as e:
        result["error"] = str(e)
    return result

def analyze_enc_file(path):
    result = {
        "type": "enc",
        "size": None,
        "entropy": None,
        "encrypted": False
    }
    try:
        with open(path, "rb") as f:
            data = f.read()
        result["size"] = len(data)
        result["entropy"] = round(calculate_entropy(data), 4)
        result["encrypted"] = result["entropy"] > 5.0
    except Exception as e:
        result["error"] = str(e)
    return result

def analyze_files(file_paths):
    report = {}
    for path in file_paths:
        filename = os.path.basename(path)
        if path.endswith(".docx"):
            report[filename] = analyze_docx(path)
        elif path.endswith(".doc"):
            report[filename] = analyze_doc(path)
        elif path.endswith(".enc") or path.endswith(".bin"):
            report[filename] = analyze_enc_file(path)
        else:
            report[filename] = {"error": "Unsupported format"}
    return report
