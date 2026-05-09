import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import (
    OCRPredictor,
    create_run_id,
    ensure_dir,
    preprocess_image,
    segment_characters,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
GENERATED_DIR = ensure_dir(STATIC_DIR / "generated")
UPLOAD_DIR = ensure_dir(GENERATED_DIR / "uploads")
PREPROCESSED_DIR = ensure_dir(GENERATED_DIR / "preprocessed")
SEGMENT_DIR = ensure_dir(GENERATED_DIR / "segments")
BOXED_DIR = ensure_dir(GENERATED_DIR / "boxed")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_FILE_SIZE_MB = 20

MODEL_PATH = Path(os.environ.get("EPIGRANET_MODEL_PATH", BASE_DIR / "models" / "epigranet_embedding_model (1).pt"))
DATASET_PATH = Path(os.environ.get("EPIGRANET_DATASET_PATH", BASE_DIR / "aug_dataset"))
CLASS_MAPPING_PATH = Path(os.environ.get("EPIGRANET_CLASS_MAPPING_PATH", BASE_DIR / "class_mapping_209 (1).json"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
app.config["JSON_AS_ASCII"] = False

_predictor = None


def get_predictor() -> OCRPredictor:
    global _predictor
    if _predictor is None:
        _predictor = OCRPredictor(MODEL_PATH, DATASET_PATH, CLASS_MAPPING_PATH)
    return _predictor


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_static_url(path: Path) -> str:
    rel = path.relative_to(STATIC_DIR).as_posix()
    return f"/static/{rel}"


@app.route("/")
def index():
    return render_template("index.html", max_size_mb=MAX_FILE_SIZE_MB)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."}), 413


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Please select an image."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format."}), 400

    run_id = create_run_id()
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = secure_filename(file.filename)

    upload_path = UPLOAD_DIR / f"{run_id}_{safe_name}"
    preprocessed_path = PREPROCESSED_DIR / f"{run_id}_preprocessed.{ext}"
    boxed_path = BOXED_DIR / f"{run_id}_boxed.{ext}"
    roi_dir = ensure_dir(SEGMENT_DIR / run_id)

    file.save(upload_path)

    pipeline_status = ["Image uploaded successfully."]
    recognized_text = ""
    confidence = 0.0
    token_predictions = []
    num_segments = 0
    warning = None

    try:
        preprocess_image(upload_path, preprocessed_path)
        pipeline_status.append("Preprocessing completed.")
    except Exception as exc:
        return jsonify({"error": f"Preprocessing failed: {exc}"}), 500

    try:
        roi_paths = segment_characters(preprocessed_path, roi_dir, boxed_path)
        num_segments = len(roi_paths)
        if not boxed_path.exists():
            boxed_path = preprocessed_path
        pipeline_status.append("Segmentation completed.")
    except Exception as exc:
        boxed_path = preprocessed_path
        roi_paths = []
        pipeline_status.append(f"Segmentation skipped: {exc}")

    try:
        predictor = get_predictor()
        result = predictor.predict_text(roi_paths, preprocessed_path)
        recognized_text = result.text
        confidence = round(result.confidence * 100, 2)
        token_predictions = result.tokens
        if not num_segments:
            num_segments = len(result.tokens)
        pipeline_status.append("OCR prediction completed.")
    except Exception as exc:
        warning = f"Prediction failed, but preprocessing/segmentation outputs are available: {exc}"
        pipeline_status.append("Prediction failed.")

    return jsonify(
        {
            "run_id": run_id,
            "recognized_text": recognized_text,
            "confidence": confidence,
            "num_segments": num_segments,
            "original_image": to_static_url(upload_path),
            "preprocessed_image": to_static_url(preprocessed_path),
            "segmented_overlay_image": to_static_url(boxed_path),
            "token_predictions": token_predictions,
            "pipeline_status": pipeline_status,
            "warning": warning,
        }
    )


@app.route("/generated/<path:filename>")
def generated(filename: str):
    return send_from_directory(GENERATED_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
