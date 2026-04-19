import gradio as gr
import numpy as np
from pathlib import Path

from config import EvaluationConfig
from embedding_model import EmbeddingModel
from utils import load_database, calculate_cosine_similarity


config = EvaluationConfig()
model = EmbeddingModel(config)
row_to_user_map, embeddings = load_database(config)


def authenticate(img: np.ndarray) -> tuple[str, float]:
    if img is None:
        raise gr.Error("No image provided.")

    db_path = Path(config.STORAGE_PATH)
    if not db_path.exists():
        raise gr.Error(
            f"Database not found at '{db_path}'. Build it first (run build_model.py)."
        )

    try:
        emb = model(img)
    except Exception:
        raise gr.Error("Could not detect a face in the image.")

    cosine_sims = calculate_cosine_similarity(emb, embeddings)
    best_idx = int(np.argmax(cosine_sims))
    best_score = float(cosine_sims[best_idx])
    best_user = row_to_user_map[best_idx]

    if best_score < config.THRESHOLD:
        return f"Authentication failed (score={best_score:.3f} < {config.THRESHOLD}).", best_score

    return f"Authenticated as user {best_user}!", best_score


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Face Authentication") as demo:
        gr.Markdown("# Face Authentication\nUpload an image to authenticate.")

        with gr.Row():
            inp = gr.Image(type="numpy", label="Input image")

        btn = gr.Button("Authenticate", variant="primary")
        out_msg = gr.Textbox(label="Result", interactive=False)
        out_score = gr.Number(label="Similarity score", precision=4, interactive=False)

        btn.click(fn=authenticate, inputs=inp, outputs=[out_msg, out_score])

    return demo


if __name__ == "__main__":
    build_ui().launch()

