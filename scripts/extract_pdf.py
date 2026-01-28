import fitz
import sys

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc, 1):
        text += f"\n\n=== PAGE {page_num} ===\n\n"
        text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_text = extract_pdf_text("docs/Mid_Sem_PPT.pdf")
    print(pdf_text)
