import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# ---------------------------------
# CONFIG (CHANGE THESE PATHS)
# ---------------------------------

MODEL_PATH = r"epigranet_embedding_model (1).pt"
DATASET_PATH = r"aug_dataset"
TEST_IMAGE_PATH = r"roi_4.png"

IMAGE_SIZE = 64
EMBED_DIM = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# ---------------------------------
# IMAGE PREPROCESSING
# ---------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# ---------------------------------
# MODEL ARCHITECTURE
# ---------------------------------

class EmbeddingNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            EMBED_DIM
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# ---------------------------------
# LOAD MODEL
# ---------------------------------

print("Loading model...")

model = EmbeddingNet().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model loaded successfully")

# ---------------------------------
# BUILD REFERENCE EMBEDDINGS
# ---------------------------------

print("Building reference embeddings...")

reference_embeddings = {}

classes = sorted(os.listdir(DATASET_PATH))

for cls in classes:

    class_path = os.path.join(DATASET_PATH, cls)

    if not os.path.isdir(class_path):
        continue

    img_files = os.listdir(class_path)

    img_path = os.path.join(class_path, img_files[0])

    img = Image.open(img_path).convert("RGB")

    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img)

    reference_embeddings[cls] = emb

print("Reference embeddings created for", len(reference_embeddings), "classes")

# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------

def predict(image_path):

    img = Image.open(image_path).convert("RGB")

    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img)

    best_class = None
    best_score = -1

    for cls, ref_emb in reference_embeddings.items():

        score = F.cosine_similarity(emb, ref_emb).item()

        if score > best_score:
            best_score = score
            best_class = cls

    return best_class, best_score


# ---------------------------------
# TEST IMAGE
# ---------------------------------

print("\nRunning prediction...")

pred_class, score = predict(TEST_IMAGE_PATH)

print("\nPrediction Result")
print("------------------")
print("Predicted Class :", pred_class)
print("Similarity Score:", score)