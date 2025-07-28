import fitz  # PyMuPDPF
import json
import re
import pandas as pd
import numpy as np
import joblib
from operator import itemgetter
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================
# 📦 ML MODEL FEATURES
# ===========================
def extract_physical_features(pdf_path):
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
# 📦 FALLBACK LOGIC
# ===========================
def fallback_parse(pdf_path):
    """Fallback parser: builds topic + headings if ML model finds nothing"""
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

    # ✅ Pick biggest bold text OR the largest font as topic
    topic_row = df.sort_values(["is_bold", "font_size"], ascending=[False, False]).iloc[0]
    topic = topic_row["text"]

    # ✅ Use all remaining lines as headings
    headings = []
    heading_counter = 1
    for idx, row in df.iterrows():
        if idx == topic_row.name:  
            continue
        headings.append({f"heading{heading_counter}": row["text"], "page": row["page"]})
        heading_counter += 1

    return {"topics": [{"topic": topic, "headings": headings}]}

# ===========================
# 📦 MAIN PROCESS FUNCTION
# ===========================
def process_pdf(pdf_path, model_path, output_json="output.json"):
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

    # 🔮 Predict with ML
    predictions = clf.predict(df[feature_cols])
    df["class"] = predictions
    filtered_df = df[df["class"].isin(["heading", "topic"])]

    output = {"topics": []}

    if not filtered_df.empty:
        print("✅ ML model produced results.")

        # ✅ Merge all 'topic' lines into one title
        topic_lines = filtered_df[filtered_df["class"] == "topic"]["text"].tolist()
        merged_topic = " ".join(topic_lines).strip() if topic_lines else "Untitled Document"

        current_topic = {"topic": merged_topic, "headings": []}

        # ✅ Add all headings
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
        print("⚠️ ML model returned no results. Using fallback...")
        output = fallback_parse(pdf_path)

    # ✅ Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output

# ===========================
# 📦 RUN SCRIPT
# ===========================
# if __name__ == "__main__":
#     pdf_path = "Sample.pdf"
#     model_path = "finalmodel.pkl"
#     result = process_pdf(pdf_path, model_path)
#     print(json.dumps(result, indent=2))
if __name__ == "__main__":
    # Paths setup
    model_path = os.path.join(BASE_DIR, "finalmodel.pkl")
    
    # Folder structure inside sample_dataset
    sample_dataset_dir = os.path.join(BASE_DIR, "sample_dataset")
    pdfs_dir = os.path.join(sample_dataset_dir, "pdfs")
    output_dir = os.path.join(sample_dataset_dir, "output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all PDFs
    for file in os.listdir(pdfs_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdfs_dir, file)
            output_json_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.json")
            print(f"\n📄 Processing: {file}")
            try:
                result = process_pdf(pdf_path, model_path, output_json=output_json_path)
                print(f"✅ Done: {output_json_path}")
            except Exception as e:
                print(f"❌ Failed to process {file}: {e}")