from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CBAM ResNet Waste Classification API",
    version="2.0.0",
    description="API for waste classification using CBAM ResNet-34 model with advanced augmentations"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Templates
templates = Jinja2Templates(directory="api/templates")

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    load_model()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application...")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        ca = self.sigmoid(out)

        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid(self.conv(x_cat))

        x = x * sa
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            layers.append(CBAM(out_channels))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet34(num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

# Configuration from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet_cbam.pth")
DEVICE = os.getenv("DEVICE", "cpu")  # Default to CPU to avoid CUDA compatibility issues

# Global model instance
model: Optional[ResNet] = None

def load_model():
    global model, DEVICE
    if model is None:
        try:
            model = ResNet34(num_classes=2)
            if os.path.exists(MODEL_PATH):
                # Try to load on specified device, fallback to CPU if CUDA error
                try:
                    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
                    model.to(DEVICE)
                    model.eval()
                    logger.info(f"Model loaded successfully from {MODEL_PATH} on {DEVICE}")
                except RuntimeError as e:
                    if "no kernel image" in str(e) or "CUDA" in str(e):
                        logger.warning(f"CUDA error loading model: {e}. Falling back to CPU.")
                        DEVICE_FALLBACK = "cpu"
                        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE_FALLBACK, weights_only=True))
                        model.to(DEVICE_FALLBACK)
                        model.eval()
                        logger.info(f"Model loaded successfully from {MODEL_PATH} on {DEVICE_FALLBACK} (fallback)")
                        DEVICE = DEVICE_FALLBACK
                    else:
                        raise e
            else:
                logger.warning(f"Model file not found at {MODEL_PATH}. Please train the model first.")
                model = None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
    return model

# Load model on startup
load_model()

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str = "CBAM ResNet-34 v2.0"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check model path and retrain if necessary.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        classes = ['Organic', 'Recyclable']
        prediction = classes[predicted.item()]

        logger.info(f"Prediction made: {prediction} with confidence {confidence.item():.4f}")
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence.item(),
            model_version="CBAM ResNet-34 v2.0"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health():
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_status": model_status,
        "device": DEVICE,
        "model_path": MODEL_PATH
    }

@app.post("/reload-model")
async def reload_model():
    """Endpoint to reload the model (useful for development)"""
    global model
    model = None
    loaded_model = load_model()
    if loaded_model:
        return {"message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")