# EpigraNet-Tamil: Tamil Epigraphical OCR

EpigraNet-Tamil is a Flask-based OCR application for recognizing Tamil epigraphical characters from inscription images. The project combines image preprocessing, contour-based character segmentation, and a PyTorch embedding model to generate predicted text along with intermediate visual outputs.

The application provides a browser UI for uploading inscription images, viewing each pipeline stage, checking confidence scores, and exporting the final OCR result as `TXT`, `CSV`, or `JSON`.

## Features

- Upload inscription images through a simple web interface
- Preprocess noisy scans using skew correction, denoising, illumination normalization, and binarization
- Segment characters and save region-of-interest crops
- Predict characters with a ResNet18-based embedding model
- Map predicted class labels to Tamil character strings using a JSON mapping file
- Return original, preprocessed, and segmented overlay images for inspection
- Export OCR output in multiple formats

## Tech Stack

- Python
- Flask
- OpenCV
- PyTorch
- Torchvision
- NumPy
- SciPy
- Pillow
- HTML, CSS, JavaScript

## Project Structure

```text
epirgranet_implementation_full/
|-- app.py
|-- pipeline.py
|-- preprocess.py
|-- segmentation.py
|-- test.py
|-- requirements.txt
|-- class_mapping_209 (1).json
|-- models/
|   `-- epigranet_embedding_model (1).pt
|-- static/
|   |-- app.js
|   |-- styles.css
|   `-- generated/
|       |-- uploads/
|       |-- preprocessed/
|       |-- boxed/
|       `-- segments/
`-- templates/
    `-- index.html
```

## How It Works

1. The user uploads an inscription image from the web UI.
2. `app.py` stores the image and creates a unique run ID.
3. `pipeline.py` preprocesses the image by correcting skew, removing long ruling lines, denoising, normalizing illumination, and binarizing the image.
4. The same pipeline segments characters using contour detection and saves each ROI.
5. The `OCRPredictor` loads the trained embedding model and compares each ROI against reference embeddings built from a labeled dataset.
6. Predicted class IDs are converted into Tamil labels using `class_mapping_209 (1).json`.
7. The app returns recognized text, confidence, segment count, and links to generated artifacts.

## Installation & Setup

### Prerequisites

- Python 3.10 or higher recommended
- `pip`

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd epirgranet_implementation_full
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure the required model assets are available

This project already includes:

- `models/epigranet_embedding_model (1).pt`
- `class_mapping_209 (1).json`

The app also expects a reference dataset directory used to build embeddings for each class. By default it looks for:

```text
aug_dataset
```

That folder is not currently present in this repository, so before running the app you should do one of the following:

1. Add an `aug_dataset/` folder at the project root.
2. Set `EPIGRANET_DATASET_PATH` to the location of your dataset.

Each class in the dataset should be stored in its own subfolder and contain at least one sample image.

### 4. Optional environment variables

You can override the default asset paths with environment variables:

```powershell
$env:EPIGRANET_MODEL_PATH="C:\path\to\epigranet_embedding_model (1).pt"
$env:EPIGRANET_DATASET_PATH="C:\path\to\aug_dataset"
$env:EPIGRANET_CLASS_MAPPING_PATH="C:\path\to\class_mapping_209 (1).json"
```

## Run the Application

```bash
python app.py
```

The Flask app starts in debug mode and is typically available at:

```text
http://127.0.0.1:5000
```

## Using the Web App

1. Open the app in your browser.
2. Upload a supported image format: `png`, `jpg`, `jpeg`, `bmp`, or `webp`.
3. Wait for the pipeline to complete.
4. Review:
   - original image
   - preprocessed image
   - segmentation overlay
   - recognized text
   - confidence score
   - pipeline log
5. Export the result as `TXT`, `CSV`, or `JSON`.

Maximum upload size is `20 MB`.

## API Endpoint

### `POST /api/predict`

Accepts a multipart form upload with the field:

```text
image
```

### Example using `curl`

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "image=@sample_inscription.jpg"
```

### Example response

```json
{
  "run_id": "a1b2c3d4e5f6",
  "recognized_text": "sample output",
  "confidence": 91.42,
  "num_segments": 12,
  "original_image": "/static/generated/uploads/a1b2c3d4e5f6_sample_inscription.jpg",
  "preprocessed_image": "/static/generated/preprocessed/a1b2c3d4e5f6_preprocessed.jpg",
  "segmented_overlay_image": "/static/generated/boxed/a1b2c3d4e5f6_boxed.jpg",
  "token_predictions": [
    {
      "label": "class label",
      "raw_label": "class_1",
      "score": 0.98
    }
  ],
  "pipeline_status": [
    "Image uploaded successfully.",
    "Preprocessing completed.",
    "Segmentation completed.",
    "OCR prediction completed."
  ],
  "warning": null
}
```

## Generated Outputs

During each run, the app stores intermediate and final artifacts under `static/generated/`:

- `uploads/` for original uploaded images
- `preprocessed/` for cleaned binary images
- `boxed/` for segmentation overlay images
- `segments/<run_id>/` for per-character ROI crops

## Notes and Limitations

- Prediction depends on the availability and quality of the reference dataset.
- The model performs nearest-reference matching using cosine similarity on learned embeddings.
- If segmentation fails, the app falls back to predicting from the full preprocessed image.
- `preprocess.py`, `segmentation.py`, and `test.py` appear to be standalone experimentation scripts, while `app.py` and `pipeline.py` drive the main application flow.

## Future Improvements

- Add the missing reference dataset structure to the repository or document it separately
- Pin dependency versions in `requirements.txt`
- Add automated tests for the Flask endpoint and OCR pipeline
- Add batch upload support and result history
- Package configuration with a `.env` file or settings module
