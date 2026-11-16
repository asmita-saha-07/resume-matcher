
import pymupdf as pm
import easyocr
import numpy as np
import streamlit as st

from PIL import Image, ImageEnhance, ImageFilter


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

            pix = page.get_pixmap(
                matrix=pm.Matrix(2, 2),
                colorspace=pm.csRGB,
                alpha=False
            )

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            # Improve contrast
            img = ImageEnhance.Contrast(img).enhance(1.8)

            # Sharpen text edges
            img = img.filter(ImageFilter.SHARPEN)

            # Moderate upscaling
            img = img.resize(
                (int(img.width * 1.3), int(img.height * 1.3)),
                Image.Resampling.LANCZOS
            )

            img_array = np.array(img)

            results = reader.readtext(
                img_array,
                detail=0,
                batch_size=1,
                workers=0,
                canvas_size=2560,
                mag_ratio=1.0
            )

            text += " ".join(results) + "\n"

    return text

