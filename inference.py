"""
inference.py — Batch runner using REDDOTRagEngine
"""
import pandas as pd
from pathlib import Path
from Orchestrator import REDDOTRagEngine


def load_text(path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def normalise_gt(raw: str) -> str:
    return "real" if raw.strip().lower() in ("true", "real") else "fake"


def run_custom_inference(test_data_root, checkpoint_path, output_csv):

    engine = REDDOTRagEngine(
        checkpoint_path = checkpoint_path,
        store_path      = "faiss_joint_store",
        k_neighbours    = 5,
        entropy_gate    = 0.50,
    )

    sample_folders = sorted([
        f for f in Path(test_data_root).iterdir() if f.is_dir()
    ])
    print(f"\nFound {len(sample_folders)} samples.")

    results = []

    for folder in sample_folders:
        try:
            result       = engine.run(
                image_input = str(folder / "image.jpg"),
                caption     = load_text(folder / "caption.txt"),
                sample_id   = folder.name,
            )
            ground_truth = normalise_gt(load_text(folder / "GT.txt"))
            pred         = result["verdict"].lower()
            correct      = (pred == ground_truth) if pred != "uncertain" else None
            status       = "✓" if correct else ("?" if correct is None else "✗")

            print(
                f"  {status} {folder.name} | GT={ground_truth} | "
                f"pred={pred} | gate={result['gate_triggered']} | "
                f"entropy={result['entropy']:.4f} | "
                f"logit={result['logit']:+.4f}"
            )

            results.append({
                "Sample_ID"        : folder.name,
                "Ground_Truth"     : ground_truth,
                "Prediction"       : pred,
                "Gate_Triggered"   : result["gate_triggered"],
                "Base_Verdict"     : result["base_verdict"].lower(),
                "Neighbour_Verdict": result["neighbour_verdict"].lower(),
                "Signals_Agree"    : result["signals_agree"],
                "Correct"          : correct,
                "Logit"            : result["logit"],
                "Prob_Fake"        : result["prob_fake"],
                "Cosine_Sim"       : result["cosine_sim"],
                "Confidence"       : result["confidence"],
                "Entropy"          : result["entropy"],
                "Latency_ms"       : result["latency_ms"],
            })

        except FileNotFoundError as e:
            print(f"  SKIP {folder.name} — {e}")
        except Exception as e:
            print(f"  ERROR {folder.name} — {e}")

    if not results:
        return

    df        = pd.DataFrame(results)
    decided   = df[df["Prediction"] != "uncertain"]
    uncertain = (df["Prediction"] == "uncertain").sum()
    gate_a    = (df["Gate_Triggered"] == "A").sum()
    gate_b    = (df["Gate_Triggered"] == "B").sum()

    acc       = round(decided["Correct"].sum() / len(decided) * 100, 2) if len(decided) else 0
    tp = ((decided["Prediction"]=="fake") & (decided["Ground_Truth"]=="fake")).sum()
    fp = ((decided["Prediction"]=="fake") & (decided["Ground_Truth"]=="real")).sum()
    fn = ((decided["Prediction"]=="real") & (decided["Ground_Truth"]=="fake")).sum()
    tn = ((decided["Prediction"]=="real") & (decided["Ground_Truth"]=="real")).sum()
    precision = round(tp/(tp+fp)*100, 2) if (tp+fp) > 0 else 0.0
    recall    = round(tp/(tp+fn)*100, 2) if (tp+fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0.0

    print("\n" + "─"*58)
    print(f"  Total          : {len(df)}")
    print(f"  Gate A (direct): {gate_a}  |  Gate B (RAG): {gate_b}")
    print(f"  Uncertain      : {uncertain}")
    print(f"  Accuracy       : {acc}%")
    print(f"  Precision      : {precision}%")
    print(f"  Recall         : {recall}%")
    print(f"  F1 Score       : {f1}%")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Mean Entropy   : {df['Entropy'].mean():.4f}")
    print(f"  Mean Cosine    : {df['Cosine_Sim'].mean():.4f}")
    print("─"*58)

    df.to_csv(output_csv, index=False)
    print(f"Report saved → {output_csv}")


if __name__ == "__main__":
    run_custom_inference(
        test_data_root  = "Test data",
        checkpoint_path = "checkpoints_pt/best_model.pt",
        output_csv      = "Final_Inference_Report.csv",
    )
