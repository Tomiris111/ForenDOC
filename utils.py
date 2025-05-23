#UTILS — utils.py
# Minimum size to consider a valid .docx file
MIN_DOCX_SIZE = 4096

# Binary header signature for .docx (ZIP file)
DOCX_SIG = b'\x50\x4B\x03\x04'

# Binary footer for .docx (End Of Central Directory)
DOCX_END = b'\x50\x4B\x05\x06'

# Binary signature for legacy .doc (OLE Compound File)
DOC_SIG = b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'

# Max chunk size when reading binary blob
BLOCK_SIZE = 8 * 1024 * 1024  # 8 MB
