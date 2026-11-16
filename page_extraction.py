import pymupdf as pm
import easyocr as e
import numpy as np
import streamlit as st

@st.cache_resource
def load_reader():
    return e.Reader(['en'], gpu=False)
    
def page_extract(resume):
    pdf_bytes = resume.getvalue()
    
    with pm.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text() + "\n"

        if not text.strip():

            reader = load_reader()
            for page in doc:
                pix = page.get_pixmap(dpi=250,colorspace=pm.csRGB,alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                results = reader.readtext(img,batch_size=1,workers=0,detail=0,canvas_size=2560,mag_ratio=1.0)

            for item in results:
                text += item[1] + " "
    return text
