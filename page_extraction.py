import pymupdf as pm
import easyocr
import numpy as np
import streamlit as st


@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)


def page_extract(resume):

    data = resume.getvalue()

    with pm.open(stream=data, filetype="pdf") as doc:

        text = ""

        for page in doc:
            text += page.get_text() + "\n"

        if not text.strip():

            reader = load_reader()

            for page in doc:

                pix = page.get_pixmap(
                    matrix=pm.Matrix(3, 3),
                    colorspace=pm.csRGB,
                    alpha=False
                )

                img = np.frombuffer(
                    pix.samples,
                    dtype=np.uint8
                )

                img = img.reshape(
                    pix.height,
                    pix.width,
                    3
                )

                result = reader.readtext(img)

                for item in result:
                    text += item[1] + " "

    return text
