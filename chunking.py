import re
import json
from pathlib import Path
import PyPDF2
import docx
from typing import List, Dict


# =========================
# PDF & DOCX EXTRACTION
# =========================

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PyPDF2.PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def extract_text_from_docx(docx_path: Path) -> str:
    doc = docx.Document(str(docx_path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# =========================
# STRUCTURAL SPLITTING
# =========================

BAB_PATTERN = re.compile(r"(BAB\s+[IVXLC]+\s*(?:\n[A-Z][A-Z\s]+)+)",re.MULTILINE)
PASAL_PATTERN = re.compile(r"(Pasal\s+\d+)")
AYAT_PATTERN = re.compile(r"(\(\d+\))")
HURUF_PATTERN = re.compile(r"(?m)^\s*([a-z])\.\s+")


def split_bab(text: str) -> Dict[str, str]:
    parts = BAB_PATTERN.split(text)
    bab_map = {}

    current_bab = None
    for part in parts:
        part = part.strip()
        if BAB_PATTERN.match(part):
            current_bab = part
            bab_map[current_bab] = ""
        elif current_bab:
            bab_map[current_bab] += part + "\n"

    return bab_map


def split_pasal(text: str) -> Dict[str, str]:
    parts = PASAL_PATTERN.split(text)
    pasal_map = {}

    current_pasal = None
    for part in parts:
        part = part.strip()
        if PASAL_PATTERN.match(part):
            current_pasal = part
            pasal_map[current_pasal] = ""
        elif current_pasal:
            pasal_map[current_pasal] += part + "\n"

    return pasal_map


def split_ayat(text: str) -> List[Dict]:
    parts = AYAT_PATTERN.split(text)
    ayat_list = []

    current_ayat = None
    for part in parts:
        part = part.strip()
        if AYAT_PATTERN.match(part):
            current_ayat = part
        elif current_ayat:
            ayat_list.append({
                "ayat": current_ayat,
                "teks": part
            })

    return ayat_list


def split_huruf(text: str) -> List[Dict]:
    parts = HURUF_PATTERN.split(text)
    hasil = []

    current_huruf = None
    for part in parts:
        part = part.strip()
        if HURUF_PATTERN.match(part):
            current_huruf = part.replace(".", "")
        elif current_huruf:
            hasil.append({
                "huruf": current_huruf,
                "teks": part
            })

    return hasil


# =========================
# MAIN CHUNKING FUNCTION
# =========================

def chunk_pertor(input_path: Path, output_path: Path):

    if input_path.suffix.lower() == ".pdf":
        full_text = extract_text_from_pdf(input_path)
    elif input_path.suffix.lower() in [".docx"]:
        full_text = extract_text_from_docx(input_path)
    else:
        raise ValueError("Unsupported file type")

    bab_dict = split_bab(full_text)

    all_chunks = []
    chunk_id = 1

    for bab, bab_text in bab_dict.items():
        pasal_dict = split_pasal(bab_text)

        for pasal, pasal_text in pasal_dict.items():
            ayat_list = split_ayat(pasal_text)

            for ayat_data in ayat_list:
                huruf_list = split_huruf(ayat_data["teks"])

                if huruf_list:
                    for h in huruf_list:
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "bab": bab,
                            "pasal": pasal,
                            "ayat": ayat_data["ayat"],
                            "huruf": h["huruf"],
                            "teks": h["teks"].strip()
                        })
                        chunk_id += 1
                else:
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "bab": bab,
                        "pasal": pasal,
                        "ayat": ayat_data["ayat"],
                        "huruf": None,
                        "teks": ayat_data["teks"].strip()
                    })
                    chunk_id += 1

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Chunking selesai. Total chunk: {len(all_chunks)}")
    print(f"✅ Output disimpan di: {output_path}")


# =========================
# CLI ENTRY POINT
# =========================

if __name__ == "__main__":
    input_file = Path("data/PERTOR_25_2023_ocr_web.pdf")
    output_file = Path("data/pertor_25_2023_chunks.json")

    chunk_pertor(input_file, output_file)
