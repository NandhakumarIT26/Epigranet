import json
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

from pipeline import OCRPredictor, ensure_dir, preprocess_image, segment_characters


BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ensure_dir(Path(os.environ.get("EPIGRANET_GENERATED_DIR", BASE_DIR / "runtime_generated")))
UPLOAD_DIR = ensure_dir(GENERATED_DIR / "uploads")
PREPROCESSED_DIR = ensure_dir(GENERATED_DIR / "preprocessed")
SEGMENT_DIR = ensure_dir(GENERATED_DIR / "segments")
BOXED_DIR = ensure_dir(GENERATED_DIR / "boxed")

DEFAULT_MODEL_PATH = BASE_DIR / "models" / "epigranet_embedding_model (1).pt"
DEFAULT_EMBEDDINGS_PATH = BASE_DIR / "models" / "reference_embeddings.pt"
DEFAULT_CLASS_MAPPING_PATH = BASE_DIR / "class_mapping_209 (1).json"
DEFAULT_DATASET_PATH = BASE_DIR / "aug_dataset"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_FILE_SIZE_MB = 20


def create_run_id() -> str:
    return uuid.uuid4().hex[:12]


def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_asset_settings() -> Dict[str, Optional[str]]:
    return {
        "hf_repo_id": os.environ.get("EPIGRANET_HF_REPO_ID"),
        "hf_revision": os.environ.get("EPIGRANET_HF_REVISION", "main"),
        "hf_token": os.environ.get("EPIGRANET_HF_TOKEN"),
        "hf_model_filename": os.environ.get("EPIGRANET_HF_MODEL_FILENAME", DEFAULT_MODEL_PATH.name),
        "hf_embeddings_filename": os.environ.get(
            "EPIGRANET_HF_EMBEDDINGS_FILENAME", DEFAULT_EMBEDDINGS_PATH.name
        ),
        "hf_class_mapping_filename": os.environ.get(
            "EPIGRANET_HF_CLASS_MAPPING_FILENAME", DEFAULT_CLASS_MAPPING_PATH.name
        ),
        "model_path": os.environ.get("EPIGRANET_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        "embeddings_path": os.environ.get("EPIGRANET_EMBEDDINGS_PATH", str(DEFAULT_EMBEDDINGS_PATH)),
        "class_mapping_path": os.environ.get("EPIGRANET_CLASS_MAPPING_PATH", str(DEFAULT_CLASS_MAPPING_PATH)),
        "dataset_path": os.environ.get("EPIGRANET_DATASET_PATH", str(DEFAULT_DATASET_PATH)),
    }


@st.cache_resource(show_spinner=False)
def resolve_runtime_assets(settings: Dict[str, Optional[str]]) -> Tuple[Path, Path, Path, Optional[Path]]:
    hf_repo_id = settings["hf_repo_id"]
    if hf_repo_id:
        from huggingface_hub import hf_hub_download

        model_path = Path(
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=str(settings["hf_model_filename"]),
                revision=str(settings["hf_revision"]),
                token=settings["hf_token"],
            )
        )
        embeddings_path = Path(
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=str(settings["hf_embeddings_filename"]),
                revision=str(settings["hf_revision"]),
                token=settings["hf_token"],
            )
        )
        class_mapping_path = Path(
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=str(settings["hf_class_mapping_filename"]),
                revision=str(settings["hf_revision"]),
                token=settings["hf_token"],
            )
        )
    else:
        model_path = Path(str(settings["model_path"]))
        embeddings_path = Path(str(settings["embeddings_path"]))
        class_mapping_path = Path(str(settings["class_mapping_path"]))

    dataset_candidate = Path(str(settings["dataset_path"]))
    dataset_path = dataset_candidate if dataset_candidate.exists() else None
    return model_path, embeddings_path, class_mapping_path, dataset_path


@st.cache_resource(show_spinner=False)
def load_predictor(settings: Dict[str, Optional[str]]) -> OCRPredictor:
    model_path, embeddings_path, class_mapping_path, dataset_path = resolve_runtime_assets(settings)
    return OCRPredictor(
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        embedding_cache_path=embeddings_path if embeddings_path.exists() else None,
        dataset_path=dataset_path,
    )


def run_ocr(uploaded_file) -> Dict[str, object]:
    filename = uploaded_file.name or "uploaded_image.png"
    if not is_allowed_file(filename):
        raise ValueError("Unsupported file format.")

    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.2f} MB. Limit is {MAX_FILE_SIZE_MB} MB.")

    run_id = create_run_id()
    ext = filename.rsplit(".", 1)[1].lower()
    safe_stem = Path(filename).stem.replace(" ", "_")

    upload_path = UPLOAD_DIR / f"{run_id}_{safe_stem}.{ext}"
    preprocessed_path = PREPROCESSED_DIR / f"{run_id}_preprocessed.png"
    boxed_path = BOXED_DIR / f"{run_id}_boxed.png"
    roi_dir = ensure_dir(SEGMENT_DIR / run_id)

    upload_path.write_bytes(file_bytes)

    preprocess_image(upload_path, preprocessed_path)
    roi_paths = segment_characters(preprocessed_path, roi_dir, boxed_path)

    predictor = load_predictor(get_asset_settings())
    result = predictor.predict_text(roi_paths, preprocessed_path)
    segments_used = len(roi_paths) if roi_paths else len(result.tokens)

    payload = {
        "run_id": run_id,
        "recognized_text": result.text,
        "confidence": round(result.confidence * 100, 2),
        "num_segments": segments_used,
        "token_predictions": result.tokens,
        "original_image": upload_path,
        "preprocessed_image": preprocessed_path,
        "segmented_overlay_image": boxed_path if boxed_path.exists() else preprocessed_path,
        "roi_paths": roi_paths,
    }
    return payload


def render_downloads(result: Dict[str, object]) -> None:
    token_predictions = result["token_predictions"]
    text_payload = str(result["recognized_text"])
    json_payload = json.dumps(
        {
            "run_id": result["run_id"],
            "recognized_text": result["recognized_text"],
            "confidence": result["confidence"],
            "num_segments": result["num_segments"],
            "token_predictions": token_predictions,
        },
        ensure_ascii=False,
        indent=2,
    )
    csv_payload = pd.DataFrame(token_predictions).to_csv(index=False)

    st.download_button("Download TXT", data=text_payload, file_name=f"{result['run_id']}.txt")
    st.download_button("Download JSON", data=json_payload, file_name=f"{result['run_id']}.json")
    st.download_button("Download CSV", data=csv_payload, file_name=f"{result['run_id']}.csv")


def main() -> None:
    st.set_page_config(page_title="EpigraNet Tamil OCR", layout="wide")
    st.title("EpigraNet Tamil OCR")
    st.caption("Streamlit frontend for the existing OCR pipeline with optional Hugging Face Hub model assets.")

    with st.sidebar:
        st.subheader("Runtime")
        settings = get_asset_settings()
        source_label = settings["hf_repo_id"] or "Local project files"
        st.write(f"Asset source: `{source_label}`")
        if settings["hf_repo_id"]:
            st.write(f"Revision: `{settings['hf_revision']}`")
        st.write(f"Upload limit: `{MAX_FILE_SIZE_MB} MB`")

    try:
        predictor = load_predictor(settings)
        st.sidebar.success(
            f"Model ready: `{predictor.model_arch}` with {len(predictor.reference_embeddings)} reference embeddings."
        )
    except Exception as exc:
        st.error(f"Unable to initialize OCR predictor: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an inscription image",
        type=sorted(ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload a Tamil inscription image to run OCR.")
        return

    preview = Image.open(BytesIO(uploaded_file.getvalue()))
    st.image(preview, caption="Uploaded image", use_container_width=True)

    if st.button("Run OCR", type="primary"):
        try:
            with st.spinner("Running preprocessing, segmentation, and OCR..."):
                result = run_ocr(uploaded_file)
        except Exception as exc:
            st.error(f"OCR failed: {exc}")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{result['confidence']:.2f}%")
        col2.metric("Segments", int(result["num_segments"]))
        col3.metric("Tokens", len(result["token_predictions"]))

        st.subheader("Recognized Text")
        st.text_area("OCR Output", value=str(result["recognized_text"]), height=120)

        image_col1, image_col2, image_col3 = st.columns(3)
        image_col1.image(str(result["original_image"]), caption="Original", use_container_width=True)
        image_col2.image(str(result["preprocessed_image"]), caption="Preprocessed", use_container_width=True)
        image_col3.image(
            str(result["segmented_overlay_image"]),
            caption="Segmentation Overlay",
            use_container_width=True,
        )

        st.subheader("Token Predictions")
        st.dataframe(pd.DataFrame(result["token_predictions"]), use_container_width=True)

        roi_paths: List[Path] = result["roi_paths"]  # type: ignore[assignment]
        if roi_paths:
            st.subheader("Character Segments")
            roi_columns = st.columns(4)
            for index, roi_path in enumerate(roi_paths):
                with roi_columns[index % len(roi_columns)]:
                    st.image(str(roi_path), caption=roi_path.name, use_container_width=True)

        st.subheader("Exports")
        render_downloads(result)


if __name__ == "__main__":
    main()
