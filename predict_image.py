import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

IMAGE_PATH = r"C:\IntelSceneModel\test_image.jpg"
MODEL_PATH = r"C:\IntelSceneModel\best_mobilenetv2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Load the saved model
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]

print("Classes:", classes)


# --------------------------------------------------
# 3. Create MobileNetV2
# --------------------------------------------------

model = models.mobilenet_v2(weights=None)

number_of_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.30),
    nn.Linear(number_of_features, 6)
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)

model.eval()


# --------------------------------------------------
# 4. Prepare the image
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = transform(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# --------------------------------------------------
# 5. Make prediction
# --------------------------------------------------

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(
        probabilities, 1
    )


# --------------------------------------------------
# 6. Display result
# --------------------------------------------------

predicted_class = classes[predicted_index.item()]

confidence_percentage = confidence.item() * 100

print("\n==============================")
print("IMAGE PREDICTION")
print("==============================")

print("Image:", IMAGE_PATH)
print("Prediction:", predicted_class)
print(f"Confidence: {confidence_percentage:.2f}%")