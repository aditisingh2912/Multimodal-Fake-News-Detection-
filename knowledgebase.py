import os
import json
import time
import numpy as np
import pandas as pd
import faiss

IMAGE_EMBEDDINGS_FILE = r"C:\Users\Dell\Multimodal-Fake-News-Detection-\VERITE_clip_image_embeddings_ViTL14.npy"
TEXT_EMBEDDINGS_FILE = r"C:\Users\Dell\Multimodal-Fake-News-Detection-\VERITE_clip_text_embeddings_ViTL14.npy"
METADATA_CSV_FILE = r"C:\Users\Dell\Multimodal-Fake-News-Detection-\VERITE.csv"
OUTPUT_PREFIX = "faiss_joint_store"

def normalise_label(raw_value) -> str:
    if pd.isna(raw_value):
        return "unknown"
    clean_str = str(raw_value).strip().lower()

    real_variants = {"true", "real", "1", "legitimate", "not fake", "original"}
    fake_variants = {"false", "fake", "0", "miscaptioned", "out-of-context", "manipulated", "altered"}

    if clean_str in real_variants: return "real"
    if clean_str in fake_variants: return "fake"
    return "unknown"


def build_joint_knowledgebase():
    start_time = time.time()
    print("=" * 60)
    print(" REDDOT - UNIFIED IMAGE-TEXT PAIR FAISS COMPILER")
    print("=" * 60)

    # ── STEP A: Load Vectors ──
    print("\n[Step 1] Loading .npy embedding arrays...")
    if not os.path.exists(IMAGE_EMBEDDINGS_FILE) or not os.path.exists(TEXT_EMBEDDINGS_FILE):
        print(" ERROR: Input embedding matrix paths are invalid.")
        return

    img_vectors = np.load(IMAGE_EMBEDDINGS_FILE).astype(np.float32)
    txt_vectors = np.load(TEXT_EMBEDDINGS_FILE).astype(np.float32)

    print(f" Raw Image Matrix: {img_vectors.shape}")
    print(f" Raw Text Matrix : {txt_vectors.shape}")

    # ── STEP B: Load Metadata & Align ──
    print(f"\n[Step 2] Aligning matrix entries with metadata sheet...")
    if not os.path.exists(METADATA_CSV_FILE):
        print("ERROR: Metadata CSV file path missing.")
        return

    df = pd.read_csv(METADATA_CSV_FILE)
    #resetiing the rows across the tables
    aligned_total = min(len(img_vectors), len(txt_vectors), len(df))
    img_vectors = img_vectors[:aligned_total]
    txt_vectors = txt_vectors[:aligned_total]
    df = df.iloc[:aligned_total].reset_index(drop=True)
    print(f"  ✓ Target synchronization dimension bound: {aligned_total} Rows.")

    # ── STEP C: L2 Normalisation (Individually) ──
    print("\n[Step 3] Normalizing structural embeddings...")

    def normalize_rows(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-9)

    img_norm = normalize_rows(img_vectors)
    txt_norm = normalize_rows(txt_vectors)

    # ── STEP D: FUSION LAYER (Concatenating Image + Text vectors side-by-side) ──
    print("\n[Step 4] Fusing Image and Text vectors into unified token structures...")
    # Har sample ab [768] + [768] = [1536] dimensions ka unified vector banega
    joint_pairs_matrix = np.hstack((img_norm, txt_norm)).astype(np.float32)

    # Final L2 normalization on the unified space to optimize cosine similarity searches
    row_norms = np.linalg.norm(joint_pairs_matrix, axis=1, keepdims=True)
    joint_pairs_matrix = joint_pairs_matrix / (row_norms + 1e-9)

    joint_dimension = joint_pairs_matrix.shape[1]  # Exactly 1536
    print(f"  ✓ Joint Vector Matrix Compiled. Shape: {joint_pairs_matrix.shape}")

    # ── STEP E: Build Single FAISS Joint Index ──
    print(f"\n[Step 5] Building 1536-Dimensional Unified FAISS IndexFlatIP...")
    joint_faiss_index = faiss.IndexFlatIP(joint_dimension)
    joint_faiss_index.add(joint_pairs_matrix)
    print(f"  ✓ Joint FAISS Nodes Registered: {joint_faiss_index.ntotal}")

    # ── STEP F: Create Metadata Tracking List ──
    print("\n[Step 6] Mapping indices back to explicit string definitions...")
    id_column = "image_id" if "image_id" in df.columns else df.columns[0]
    caption_column = "caption" if "caption" in df.columns else "text"
    label_column = "label" if "label" in df.columns else df.columns[-1]

    json_metadata_store = []
    for idx, row in df.iterrows():
        clean_verdict = normalise_label(row[label_column])
        cosine_sim_value = float(np.dot(img_norm[idx], txt_norm[idx]))

        json_metadata_store.append({
            "matrix_row_id": idx,  # Direct link to FAISS location row index
            "sample_id": str(row.get(id_column, idx)),
            "verdict": clean_verdict,
            "caption": str(row.get(caption_column, ""))[:120],
            "internal_clip_sim": round(cosine_sim_value, 4)
        })

    # ── STEP G: Save to Disk ──
    print("\n[Step 7] Exporting single master index configuration to folder paths...")
    faiss.write_index(joint_faiss_index, f"{OUTPUT_PREFIX}.index")

    with open(f"{OUTPUT_PREFIX}_meta.json", "w") as json_file:
        json.dump(json_metadata_store, json_file, indent=2)

    with open(f"{OUTPUT_PREFIX}_config.json", "w") as config_file:
        json.dump({"dim": joint_dimension, "n_entries": len(json_metadata_store)}, config_file)

    total_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 60)
    print(f"SUCCESS: Single Unified Image-Text Pair FAISS Index Created!")
    print(f" Output files: '{OUTPUT_PREFIX}.index' & '{OUTPUT_PREFIX}_meta.json'")
    print(f"Total Execution Time: {total_time} seconds")
    print("=" * 60)


if __name__ == "__main__":
    build_joint_knowledgebase()