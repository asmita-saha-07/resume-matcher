import pymupdf as pm
import easyocr as e
import fitz
import numpy as np

def page_extract(resume):

    data = resume.getvalue()       

    with pm.open(stream=data, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text() + "\n"

    if not text.strip():
        file= e.Reader(['en'])
        pages= pm.open(stream=data, filetype="pdf")

        text= ""

        for pg in pages:
            pix= pg.get_pixmap( matrix=pm.Matrix(4, 4),colorspace=pm.csRGB,alpha=False)
            img= np.frombuffer(pix.samples, dtype=np.uint8)
            img= img.reshape(pix.height, pix.width, 3)

            result= file.readtext(img)

            for data in result:
                text += data[1] + " "
    return text