# slot_image_recovery.py

import os
import zipfile

def extract_images(docx_path, output_folder="recovered_images"):
    """
    Extract all embedded images from a single .docx file 
    and save them to the specified output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with zipfile.ZipFile(docx_path, 'r') as docx:
            for file in docx.namelist():
                if file.startswith("word/media/") and file.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                    filename = os.path.basename(file)
                    output_path = os.path.join(output_folder, filename)

                    # Ensure unique filenames to avoid overwriting
                    base, ext = os.path.splitext(filename)
                    i = 1
                    while os.path.exists(output_path):
                        filename = f"{base}_{i}{ext}"
                        output_path = os.path.join(output_folder, filename)
                        i += 1

                    with open(output_path, "wb") as img:
                        img.write(docx.read(file))
    except Exception as e:
        print(f"[!] Failed to extract from {docx_path}: {e}")


def extract_images_from_all_docx(docx_folder="recovered_docs_from_fragment", output_folder="recovered_images"):
    """
    Loop through all .docx files in the given folder and extract images from each.
    """
    if not os.path.exists(docx_folder):
        print(f"[!] Folder {docx_folder} does not exist.")
        return

    count = 0
    for file in os.listdir(docx_folder):
        if file.lower().endswith(".docx"):
            full_path = os.path.join(docx_folder, file)
            extract_images(full_path, output_folder)
            count += 1

    print(f"[✓] Extracted images from {count} .docx files into '{output_folder}'")
