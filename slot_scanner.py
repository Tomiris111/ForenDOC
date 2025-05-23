import os
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

DOCX_SIGNATURE = b'PK\x03\x04'
OUTPUT_FOLDER = "fragments"
FRAGMENT_SIZE = 20 * 1024 * 1024  # 20 MB per fragment

def _extract_and_write(args):
    """
    Worker function: open the dump and write all fragments for a chunk of offsets.
    """
    dump_path, offsets, part_idx = args
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_file = os.path.join(OUTPUT_FOLDER, f"fragment_part{part_idx+1}.bin")
    with open(dump_path, "rb") as f, open(output_file, "wb") as out:
        for offset in offsets:
            f.seek(offset)
            chunk = f.read(FRAGMENT_SIZE)
            out.write(chunk)
    return len(offsets)

def scan_and_create_accurate_fragment(dump_path):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    signature_offsets = []
    read_size = 1024 * 1024  # 1 MB
    total_bytes_read = 0

    # --- Phase 1: scan for DOCX signatures ---
    with open(dump_path, "rb") as f:
        while True:
            block = f.read(read_size)
            if not block:
                break
            offset_in_block = 0
            while True:
                idx = block.find(DOCX_SIGNATURE, offset_in_block)
                if idx == -1:
                    break
                signature_offsets.append(total_bytes_read + idx)
                offset_in_block = idx + len(DOCX_SIGNATURE)
            total_bytes_read += len(block)

    count = len(signature_offsets)
    print(f"[✓] Found {count} DOCX signatures.")

    if count == 0:
        print("[!] No DOCX signatures found.")
        return 0, total_bytes_read // read_size

    # --- Phase 2: parallel extraction ---
    cpu_count = multiprocessing.cpu_count()
    # limit to 70% of available cores, at least 1
    num_workers = max(1, int(cpu_count * 0.7))
    # determine chunk size per worker
    chunk_size = math.ceil(count / num_workers)
    # split offsets into chunks
    offset_chunks = [
        signature_offsets[i:i + chunk_size]
        for i in range(0, count, chunk_size)
    ]

    print(f"[✓] Extracting fragments using {len(offset_chunks)} workers...")

    # prepare arguments for each worker
    tasks = [
        (dump_path, chunk, idx)
        for idx, chunk in enumerate(offset_chunks)
    ]

    # run workers
    with ProcessPoolExecutor(max_workers=len(offset_chunks)) as executor:
        results = executor.map(_extract_and_write, tasks)

    total_fragments = sum(results)
    print(f"[✓] Extracted {total_fragments} fragments into '{OUTPUT_FOLDER}'.")
    return count, total_bytes_read // read_size

