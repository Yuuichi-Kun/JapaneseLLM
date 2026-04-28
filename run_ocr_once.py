# run_ocr_once.py
import pytesseract
from pdf2image import convert_from_path
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

PDF_PATH = r"C:\Users\mhana\Downloads\Buku Materi Minna no Nihongo I (2nd edition).pdf"
OUTPUT_PATH = r"C:\Users\mhana\Documents\minna_text.txt"

print("Mengkonversi PDF ke gambar... (ini butuh waktu)")
images = convert_from_path(PDF_PATH, dpi=200, poppler_path=POPPLER_PATH)
print(f"Total halaman: {len(images)}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, img in enumerate(images):
        print(f"OCR halaman {i+1}/{len(images)}...", end="\r")
        
        text = pytesseract.image_to_string(
            img,
            lang="jpn+ind",
            config="--psm 3"
        )
        
        cleaned = text.replace("www.japandaisuki.com", "").strip()
        if cleaned:
            f.write(f"\n\n=== HALAMAN {i+1} ===\n\n")
            f.write(cleaned)

print(f"\nSelesai! Teks tersimpan di: {OUTPUT_PATH}")
print(f"Ukuran file: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")