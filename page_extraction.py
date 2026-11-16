
import pymupdf as pm
import easyocr
import numpy as np
import streamlit as st
import cv2


@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)


def page_extract(resume):

    pdf_bytes = resume.getvalue()
    reader = None
    text = ""

    with pm.open(stream=pdf_bytes, filetype="pdf") as doc:

        for page in doc:
            page_text = page.get_text()

            if page_text.strip():
                text += page_text + "\n"
                continue

            if reader is None:
                reader = load_reader()

            # Render at 2x instead of 4x
            pix = page.get_pixmap(
                matrix=pm.Matrix(2, 2),
                colorspace=pm.csRGB,
                alpha=False
            )

            img = np.frombuffer(
                pix.samples, dtype=np.uint8
            ).reshape(pix.height, pix.width, 3)

            # Improve contrast
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            enhanced = clahe.apply(gray)

            # Upscale moderately, not to 4x
            enhanced = cv2.resize(
                enhanced,
                None,
                fx=1.3,
                fy=1.3,
                interpolation=cv2.INTER_CUBIC
            )

            # detail=0 returns strings, not tuples
            results = reader.readtext(
                enhanced,
                detail=0,
                batch_size=1,
                workers=0,
                canvas_size=2560,
                mag_ratio=1.0
            )

            text += " ".join(results) + "\n"

    return text
