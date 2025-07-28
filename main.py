import os
import json
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# Load local model once
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text("blocks") for page in doc]

def clean_blocks(blocks):
    return " ".join(block[4] for block in blocks if block[4].strip())

def chunk_text(text, chunk_size=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_relevant_chunks(chunks, persona_embedding, top_k=5):
    if len(chunks) == 0:
        return []

    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(chunk_embeddings, persona_embedding)

    if similarities.dim() > 1:
        similarities = similarities.squeeze()

    if similarities.dim() == 0:
        return []

    top_indices = similarities.argsort(descending=True)[:top_k]
    return [(i.item(), chunks[i], similarities[i].item()) for i in top_indices]

def validate_input(config, collection_dir):
    required_keys = ["persona", "job", "documents"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"❌ Missing keys in {collection_dir}/challenge1b_input.json: {', '.join(missing_keys)}")

def process_collection(collection_dir):
    input_path = os.path.join(collection_dir, "challenge1b_input.json")
    pdf_dir = os.path.join(collection_dir, "PDFs")
    output_path = os.path.join(collection_dir, "challenge1b_output.json")

    with open(input_path, 'r') as f:
        config = json.load(f)

    validate_input(config, collection_dir)

    persona = config["persona"]
    job = config["job"]
    raw_docs = config["documents"]
    documents = [doc if isinstance(doc, str) else doc.get("filename", "") for doc in raw_docs]
    documents = [doc for doc in documents if doc]

    persona_task = f"{persona}. Task: {job}"
    persona_embedding = model.encode(persona_task, convert_to_tensor=True)

    output = {
        "metadata": {
            "documents": documents,
            "persona": persona,
            "job": job,
            "timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for doc_file in documents:
        full_pdf_path = os.path.join(pdf_dir, doc_file)
        if not os.path.exists(full_pdf_path):
            print(f"⚠️ File not found: {full_pdf_path}. Skipping.")
            continue

        all_page_blocks = extract_text_by_page(full_pdf_path)

        for page_number, blocks in enumerate(all_page_blocks, start=1):
            text = clean_blocks(blocks)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            top_chunks = get_relevant_chunks(chunks, persona_embedding)
            if not top_chunks:
                continue

            for idx, content, score in top_chunks:
                section_title = content.split(".")[0][:60]
                output["extracted_sections"].append({
                    "document": doc_file,
                    "page_number": page_number,
                    "section_title": section_title,
                    "importance_rank": round(score, 4)
                })
                output["subsection_analysis"].append({
                    "document": doc_file,
                    "page_number": page_number,
                    "refined_text": content
                })

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Processed {collection_dir} and saved output")

def main():
    base_dir = Path(".")
    for collection in sorted(base_dir.glob("Collection*/")):
        if collection.is_dir():
            try:
                process_collection(str(collection))
            except KeyError as e:
                print(e)
            except Exception as e:
                print(f"❌ Error processing {collection}: {e}")

if __name__ == "__main__":
    main()
