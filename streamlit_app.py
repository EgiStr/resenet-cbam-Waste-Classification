import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import model
try:
    from models.cbam_resnet import ResNet34
except ImportError:
    # Fallback if running from different directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
    from cbam_resnet import ResNet34

# Page configuration
st.set_page_config(
    page_title="AI Klasifikasi Sampah",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = "models/resnet_cbam.pth"
CLASSES = ['Organik', 'Daur Ulang']

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .organik {
        background-color: #e8f5e9;
        border: 2px solid #4CAF50;
        color: #2e7d32;
    }
    .daur_ulang {
        background-color: #e3f2fd;
        border: 2px solid #2196F3;
        color: #1565c0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load and cache the model to avoid reloading on every interaction"""
    try:
        # Menggunakan CPU secara default untuk stabilitas dan menghindari error inkompatibilitas CUDA
        # Error 'no kernel image' biasanya terjadi karena ketidakcocokan versi PyTorch dan Driver GPU
        device = torch.device("cpu")
        
        model = ResNet34(num_classes=2)
        
        if os.path.exists(MODEL_PATH):
            # Load weights
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info(f"Model loaded on {device}")
            return model, device
        else:
            st.error(f"Model file not found at {MODEL_PATH}")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/recycle-sign--v1.png", width=100)
        st.title("Klasifikasi Sampah")
        st.markdown("---")
        st.markdown("""
        ### Tentang
        Aplikasi ini menggunakan model Deep Learning **CBAM-ResNet34** untuk mengklasifikasikan sampah menjadi:
        
        * 🥬 **Organik**
        * ♻️ **Daur Ulang**
        
        Model ini memanfaatkan **Convolutional Block Attention Modules (CBAM)** untuk fokus pada fitur yang relevan.
        """)
        st.markdown("---")
        st.info("Unggah gambar untuk memulai.")
        
        # Tampilkan info perangkat
        if 'device' in locals() and device is not None:
            st.caption(f"Perangkat: {str(device).upper()}")

    # Main Content
    st.title("♻️ Klasifikasi Sampah Cerdas")
    st.markdown("Unggah gambar sampah untuk diklasifikasikan secara otomatis.")

    # Model Loading
    with st.spinner("Memuat Model AI..."):
        model, device = load_model()

    if model is None:
        st.warning("⚠️ Model tidak dapat dimuat. Silakan periksa apakah file model ada.")
        return

    # File Uploader
    uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Gambar Masukan")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True, caption="Gambar yang Diunggah")

        with col2:
            st.subheader("Hasil Analisis")
            
            if st.button("Klasifikasi Sampah"):
                with st.spinner("Menganalisis gambar..."):
                    try:
                        # Preprocess
                        input_tensor = preprocess_image(image).to(device)
                        
                        # Inference
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                        
                        # Result
                        class_idx = predicted.item()
                        class_name = CLASSES[class_idx]
                        conf_score = confidence.item()
                        
                        # Display Result
                        css_class = class_name.lower().replace(" ", "_")
                        st.markdown(f"""
                            <div class="prediction-box {css_class}">
                                <h2>{class_name}</h2>
                                <p>Keyakinan: {conf_score:.2%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar for confidence
                        st.progress(conf_score)
                        
                        # Detailed probabilities
                        st.markdown("### Probabilitas Kelas")
                        for idx, name in enumerate(CLASSES):
                            prob = probabilities[0][idx].item()
                            st.write(f"**{name}**: {prob:.2%}")
                            st.progress(prob)
                            
                        # Actionable advice
                        st.markdown("### 💡 Saran Pembuangan")
                        if class_name == 'Organik':
                            st.info("Item ini sebaiknya dibuat kompos atau dibuang ke tempat sampah organik.")
                        else:
                            st.success("Item ini dapat didaur ulang. Pastikan bersih dan kering sebelum didaur ulang.")
                            
                    except Exception as e:
                        st.error(f"Kesalahan selama klasifikasi: {str(e)}")
                        logger.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
