import numpy as np, gc, easyocr, streamlit as st
import pymupdf as pm  # or: import fitz as pm

@st.cache_resource
def get_reader():
    return easyocr.Reader(['en'], gpu=False)

def page_extract(pdf_bytes):
    doc = pm.open(stream=pdf_bytes, filetype="pdf")
    reader = None
    out = []
    for page in doc:
        # 1) try real text first (works for most resumes)
        text = page.get_text().strip()
        if len(text) > 30:
            out.append(text)
            continue

        # 2) OCR fallback, with the image width capped
        if reader is None:
            reader = get_reader()
        zoom = min(1600 / page.rect.width, 3)
        pix = page.get_pixmap(matrix=pm.Matrix(zoom, zoom),
                              colorspace=pm.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        out.append(" ".join(reader.readtext(img, detail=0, paragraph=True)))
        del pix, img
        gc.collect()
    return "\n".join(out)

