from pathlib import Path

from pypdf import PdfReader

# file_path = Path("data/documents/fastapi.txt")

# text = file_path.read_text(encoding="utf-8")

# print("Document loaded successfully.")
# print("\nDocument content:")
# print(text)



def load_text_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(encoding="utf-8")


# text = load_text_file("data/documents/fastapi.txt")

# print(text)

def load_pdf_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    reader = PdfReader(str(path))
    
    pages = []
    
    for page in reader.pages:
        
        text = page.extract_text()
        
        if text:
            pages.append(text)
            
    return "\n".join(pages)

# text = load_pdf_file("data/documents/docker.pdf")

# print(text[:3000])
