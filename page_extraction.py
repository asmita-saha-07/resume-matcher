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
                    matrix=pm.Matrix(2,2),
                    colorspace=pm.csRGB,
                    alpha=False
                )

                img = np.frombuffer(
                    pix.samples, dtype=np.uint8
                ).reshape(pix.height, pix.width, 3)

                st.write("Step 5: Running OCR")

                try:

                    results = reader.readtext(
                            img,
                            batch_size=1,
                            workers=0,
                            detail=0,
                            canvas_size=2560,
                            mag_ratio=1.0
                        )

                    st.write(f"OCR Successful")

                    for item in results:
                        text += item[1] + " "

                except Exception:
                    st.write(f"OCR failed")

    st.write("Step 6: Extraction completed")

    return text
