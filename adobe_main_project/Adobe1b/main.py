import os
import json
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===========================
# 📦 1️⃣ ML MODEL FEATURES EXTRACTION
# ===========================
def extract_physical_features(pdf_path):
    """Extracts layout-based features for ML classification."""
    doc = fitz.open(pdf_path)
    records = []
    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        prev_y = None
        line_idx = 0
        page_width = page.rect.width
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                text = " ".join([span["text"].strip() for span in line["spans"] if span["text"].strip()])
                if not text:
                    continue
                sizes = [span["size"] for span in line["spans"]]
                bold_flags = [span["flags"] & 2 > 0 for span in line["spans"]]
                x0, y0, x1, y1 = line["bbox"]
                font_mean = np.mean(sizes)
                gap_before = 0 if prev_y is None else y0 - prev_y
                prev_y = y1
                records.append({
                    "text": text,
                    "page_number": page_number,
                    "line_idx_on_page": line_idx,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "bbox_width": x1 - x0,
                    "bbox_height": y1 - y0,
                    "font_size_min": np.min(sizes),
                    "font_size_max": np.max(sizes),
                    "font_size_mean": font_mean,
                    "font_size_median": np.median(sizes),
                    "any_bold": int(any(bold_flags)),
                    "all_bold": int(all(bold_flags)),
                    "gap_before": gap_before,
                    "indent_abs": abs(x0),
                    "indent_norm": x0 / page_width if page_width else 0,
                    "word_count": len(text.split()),
                    "caps_ratio": sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
                    "starts_with_number": int(text.lstrip()[0].isdigit()) if len(text.strip()) > 0 else 0,
                    "starts_with_alpha_marker": int(text.lstrip()[0].isalpha() and text.lstrip()[1:2] in [".", ")"]) if len(text.strip()) > 1 else 0,
                })
                line_idx += 1
    df = pd.DataFrame(records)
    df["gap_after"] = df.groupby("page_number")["gap_before"].shift(-1).fillna(0)
    for page in df["page_number"].unique():
        mask = df["page_number"] == page
        font_means = df.loc[mask, "font_size_mean"]
        df.loc[mask, "font_rank"] = font_means.rank(method='min', ascending=False)
        df.loc[mask, "normalized_text_size_with_page"] = font_means / font_means.max()
    return df


# ===========================
# 📦 2️⃣ FALLBACK PARSER
# ===========================
def fallback_parse(pdf_path):
    """Fallback parser if ML finds no topics or headings."""
    doc = fitz.open(pdf_path)
    blocks = []
    for page_number, page in enumerate(doc, start=1):
        for b in page.get_text("dict")["blocks"]:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                text = " ".join(span["text"].strip() for span in line["spans"] if span["text"].strip())
                if not text:
                    continue
                sizes = [span["size"] for span in line["spans"]]
                bold_flags = [span["flags"] & 2 > 0 for span in line["spans"]]
                blocks.append({
                    "text": text,
                    "page": page_number,
                    "font_size": np.mean(sizes),
                    "is_bold": int(any(bold_flags))
                })

    df = pd.DataFrame(blocks)
    if df.empty:
        return {"topics": [{"topic": "Untitled Document", "headings": []}]}

    topic_row = df.sort_values(["is_bold", "font_size"], ascending=[False, False]).iloc[0]
    topic = topic_row["text"]

    headings = []
    heading_counter = 1
    for idx, row in df.iterrows():
        if idx == topic_row.name:
            continue
        headings.append({f"heading{heading_counter}": row["text"], "page": row["page"]})
        heading_counter += 1

    return {"topics": [{"topic": topic, "headings": headings}]}


# ===========================
# 📦 3️⃣ TOPIC & HEADING EXTRACTOR
# ===========================
def process_pdf(pdf_path, model_path):
    clf = joblib.load(model_path)
    df = extract_physical_features(pdf_path)

    feature_cols = [
        "page_number", "line_idx_on_page", "x0", "y0", "x1", "y1",
        "bbox_width", "bbox_height", "font_size_min", "font_size_max", "font_size_mean", "font_size_median",
        "any_bold", "all_bold", "gap_before", "gap_after",
        "indent_abs", "indent_norm", "word_count", "caps_ratio",
        "starts_with_number", "starts_with_alpha_marker",
        "font_rank", "normalized_text_size_with_page"
    ]

    predictions = clf.predict(df[feature_cols])
    df["class"] = predictions
    filtered_df = df[df["class"].isin(["heading", "topic"])]

    filename = os.path.basename(pdf_path)
    output = {"document": filename, "topics": []}

    if not filtered_df.empty:
        topic_lines = filtered_df[filtered_df["class"] == "topic"]["text"].tolist()
        merged_topic = " ".join(topic_lines).strip() if topic_lines else "Untitled Document"

        current_topic = {"topic": merged_topic, "headings": []}
        heading_counter = 1
        for _, row in filtered_df.iterrows():
            if row["class"] == "heading":
                current_topic["headings"].append({
                    f"heading{heading_counter}": row["text"],
                    "page": row["page_number"]
                })
                heading_counter += 1
        output["topics"].append(current_topic)
    else:
        output["topics"] = fallback_parse(pdf_path)["topics"]

    return output


# ===========================
# 📦 4️⃣ PERSONA RANKING
# ===========================
def rank_sections(all_extracted, persona, job_to_be_done):
    persona_query = f"{persona} - {job_to_be_done}"
    def clean_text(text):
        return text.lower()

    all_headings = []
    heading_map = []

    for doc in all_extracted["documents"]:
        doc_name = doc.get("document", "Unknown.pdf")
        for topic in doc.get("topics", []):
            for heading in topic.get("headings", []):
                for key, text in heading.items():
                    if key.startswith("heading"):
                        page = heading.get("page", None)
                        all_headings.append(clean_text(text))
                        heading_map.append({
                            "document": doc_name,
                            "page": page,
                            "section_title": text
                        })

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([clean_text(persona_query)] + all_headings)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    for idx, score in enumerate(cosine_scores):
        heading_map[idx]["importance_rank"] = float(score)

    ranked = sorted(heading_map, key=lambda x: x["importance_rank"], reverse=True)
    return {
        "metadata": {
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processed_timestamp": datetime.now().isoformat(),
            "total_documents": len(all_extracted["documents"])
        },
        "ranked_sections": ranked
    }


# ===========================
# 📦 5️⃣ SUBSECTION EXTRACTION (CONFIDENCE THRESHOLD)
# ===========================
def extract_subsections(pdf_path, heading_text, page_number, context_lines=3):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    blocks = page.get_text("blocks")

    extracted_texts = []
    for idx, block in enumerate(blocks):
        if heading_text in block[4]:
            extracted_texts.append(block[4])
            for i in range(1, context_lines + 1):
                if idx + i < len(blocks):
                    extracted_texts.append(blocks[idx + i][4])
            break
    return " ".join(extracted_texts).strip()


def enrich_with_subsections(topics_json, persona_ranked_json, pdf_folder, min_confidence=0.2):
    """Extracts only those sections whose importance_rank is >= min_confidence."""
    ranked_sections = persona_ranked_json.get("ranked_sections", [])
    filtered_sections = [sec for sec in ranked_sections if sec["importance_rank"] >= min_confidence]

    enriched_data = {
        "metadata": {
            "processed_on": datetime.now().isoformat(),
            "total_documents": len(topics_json["documents"]),
            "sections_considered": len(filtered_sections),
            "min_confidence_threshold": min_confidence
        },
        "extracted_sections": []
    }

    seen_refined_texts = set()

    for section in filtered_sections:
        pdf_file = section["document"]
        heading_text = section["section_title"]
        page = section.get("page", 1)
        pdf_path = os.path.join(pdf_folder, pdf_file)

        try:
            refined_text = extract_subsections(pdf_path, heading_text, page)
        except Exception as e:
            refined_text = f"[Error extracting text: {e}]"

        if refined_text not in seen_refined_texts:
            seen_refined_texts.add(refined_text)
            enriched_data["extracted_sections"].append({
                "document": pdf_file,
                "heading": heading_text,
                "page": page,
                "importance_rank": section["importance_rank"],
                "refined_text": refined_text
            })

    return enriched_data


# ===========================
# 📦 6️⃣ MAIN PIPELINE
# ===========================
def main():
    input_dir = "/app/input"
    output_dir = "/app/output"
    model_path = "/app/finalmodel.pkl"

    persona = os.environ.get("PERSONA", "Traveller")
    job_to_be_done = os.environ.get("JOB_TO_BE_DONE", "visit places and eat french cuisine")
    min_confidence = float(os.environ.get("MIN_CONFIDENCE", 0.2))  # ✅ confidence threshold

    # 1️⃣ Process all PDFs to extract topics/headings
    all_results = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file)
            print(f"Processing: {file}")
            result = process_pdf(pdf_path, model_path)
            all_results.append(result)

    topics_headings_json = {"documents": all_results}
    with open(os.path.join(output_dir, "topics_headings.json"), "w", encoding="utf-8") as f:
        json.dump(topics_headings_json, f, indent=2, ensure_ascii=False)

    # 2️⃣ Persona ranking
    persona_ranked_json = rank_sections(topics_headings_json, persona, job_to_be_done)
    with open(os.path.join(output_dir, "persona_ranked.json"), "w", encoding="utf-8") as f:
        json.dump(persona_ranked_json, f, indent=2, ensure_ascii=False)

    # 3️⃣ Subsection analysis using ONLY sections with confidence >= min_confidence
    final_subsections_json = enrich_with_subsections(topics_headings_json, persona_ranked_json, input_dir, min_confidence)
    with open(os.path.join(output_dir, "final_subsections.json"), "w", encoding="utf-8") as f:
        json.dump(final_subsections_json, f, indent=2, ensure_ascii=False)

    print(f"✅ All processing complete. Sections with confidence >= {min_confidence} extracted.")


# ===========================
# 📦 RUN APP
# ===========================
if __name__ == "__main__":
    main()
