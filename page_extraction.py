import pymupdf as pm
import easyocr as e
import numpy as np
import streamlit as st

@st.cache_resource
def load_reader():
    return e.Reader(['en'], gpu=False)
    
def page_extract(resume):

    st.write("Step 1: Reading PDF")

    pdf_bytes = resume.getvalue()

    with pm.open(stream=pdf_bytes, filetype="pdf") as doc:

        text = ""

        for page in doc:
            text += page.get_text() + "\n"

        st.write("Step 2: PDF text extraction completed")

        if not text.strip():

            st.write("Step 3: Initializing EasyOCR")

            reader = load_reader()

            st.write("Step 4: EasyOCR initialized")

            for page in doc:

                pix = page.get_pixmap(
                    matrix=pm.Matrix(4, 4),
                    colorspace=pm.csRGB,
                    alpha=False
                )

                img = np.frombuffer(
                    pix.samples, dtype=np.uint8
                ).reshape(pix.height, pix.width, 3)

                st.write("Step 5: Running OCR")

                results = reader.readtext(img)

                for item in results:
                    text += item[1] + " "

    st.write("Step 6: Extraction completed")

    return text
