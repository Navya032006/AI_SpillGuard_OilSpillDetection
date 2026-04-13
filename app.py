import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# ============ Model — EXACT match to notebook ============
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1       = double_conv(in_channels, 64)
        self.pool1      = nn.MaxPool2d(2)
        self.enc2       = double_conv(64, 128)
        self.pool2      = nn.MaxPool2d(2)
        self.enc3       = double_conv(128, 256)
        self.pool3      = nn.MaxPool2d(2)
        self.enc4       = double_conv(256, 512)
        self.pool4      = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)
        self.up4        = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4       = double_conv(1024, 512)
        self.up3        = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3       = double_conv(512, 256)
        self.up2        = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2       = double_conv(256, 128)
        self.up1        = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1       = double_conv(128, 64)
        self.final      = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pth')
    if not os.path.exists(model_path):
        st.error("❌ best_model.pth not found. Place it in the same folder as app.py")
        st.stop()
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess(img):
    # ✅ FIXED: use cv2 BGR exactly like training (OilSpillDataset.__getitem__)
    # Convert PIL → numpy → cv2 BGR → resize → normalize
    img_np  = np.array(img)                              # RGB numpy
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)   # RGB → BGR (matches cv2.imread)
    img_bgr = cv2.resize(img_bgr, (256, 256))            # resize
    img_bgr = img_bgr.astype(np.float32) / 255.0        # normalize [0,1]
    return img_bgr                                        # shape (256,256,3) BGR

def predict(model, img_bgr):
    # HWC → CHW tensor exactly like training
    tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor).squeeze().numpy()
    return pred

def bgr_to_rgb(img_bgr):
    # For display only — convert back to RGB
    return cv2.cvtColor((img_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

# ============ Page Config ============
st.set_page_config(
    page_title="AI SpillGuard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    header[data-testid="stHeader"] {
        background: #0a0e1a !important;
        border-bottom: 1px solid #1e3a5f !important;
    }
    [data-testid="stToolbar"]    { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer    { visibility: hidden !important; }
    .stDeployButton { display: none !important; }

    html, body { background: #0a0e1a !important; }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%) !important;
    }
    .main, .block-container {
        background: transparent !important;
        padding-top: 1rem !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a0e1a 100%) !important;
        border-right: 1px solid #1e3a5f !important;
    }
    section[data-testid="stSidebar"] > div { background: transparent !important; }
    section[data-testid="stSidebar"] * { color: #ccd6f6; }

    [data-testid="stFileUploader"] {
        background: #0d1b2a !important;
        border-radius: 12px !important;
        border: 2px dashed #1e3a5f !important;
        padding: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #0a1628 !important;
        border: none !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small { color: #8899aa !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #1e3a5f !important;
        color: #00d4ff !important;
        border: 1px solid #00d4ff44 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderFile"] {
        background: #0d1b2a !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderFile"] span { color: #ccd6f6 !important; }

    .stMarkdown p, .stMarkdown div, p, span, label,
    [data-testid="stMarkdownContainer"] p { color: #ccd6f6 !important; }
    h1, h2, h3 { color: #00d4ff !important; }
    .stSlider label, .stSlider p { color: #ccd6f6 !important; }
    hr { border-color: #1e3a5f !important; }

    .hero {
        background: linear-gradient(135deg, #0d1b2a, #1a2744);
        border: 1px solid #1e3a5f;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 0 40px #00d4ff22;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #0099ff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero-sub  { color: #8899aa !important; font-size: 1.1rem; margin-bottom: 20px; }
    .hero-badges { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
    .badge {
        background: #1e3a5f; color: #00d4ff !important;
        padding: 5px 15px; border-radius: 20px;
        font-size: 0.8rem; border: 1px solid #00d4ff44;
    }

    .upload-card {
        background: linear-gradient(135deg, #0d1b2a, #1a2744);
        border: 2px dashed #1e3a5f; border-radius: 15px;
        padding: 25px; text-align: center; transition: all 0.3s;
    }
    .upload-card:hover { border-color: #00d4ff; box-shadow: 0 0 20px #00d4ff22; }

    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
    .metric-card {
        background: linear-gradient(135deg, #0d1b2a, #1a2744);
        border-radius: 15px; padding: 20px; text-align: center;
        border: 1px solid #1e3a5f; transition: all 0.3s;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px #00d4ff22; }
    .metric-val { font-size: 2.2rem; font-weight: 700; color: #00d4ff !important; }
    .metric-lbl { color: #8899aa !important; font-size: 0.85rem; margin-top: 5px; }

    .alert-danger {
        background: linear-gradient(135deg, #2d0a0a, #1a0505);
        border: 1px solid #ff4444; border-radius: 15px;
        padding: 20px; text-align: center; box-shadow: 0 0 20px #ff444422;
    }
    .alert-danger-title { color: #ff4444 !important; font-size: 1.4rem; font-weight: 700; }
    .alert-safe {
        background: linear-gradient(135deg, #0a2d0a, #051a05);
        border: 1px solid #00cc66; border-radius: 15px;
        padding: 20px; text-align: center; box-shadow: 0 0 20px #00cc6622;
    }
    .alert-safe-title { color: #00cc66 !important; font-size: 1.4rem; font-weight: 700; }

    .result-header {
        color: #00d4ff !important; font-size: 1.3rem; font-weight: 600;
        margin: 20px 0 10px 0; padding-bottom: 8px; border-bottom: 1px solid #1e3a5f;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0066cc, #00aaff) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 12px 30px !important;
        font-weight: 600 !important; font-size: 1rem !important;
        width: 100% !important; transition: all 0.3s !important;
        box-shadow: 0 4px 15px #0066cc44 !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px #0066cc66 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============ Sidebar ============
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0;'>
        <div style='font-size:3rem'>🛢️</div>
        <div style='color:#00d4ff; font-size:1.3rem; font-weight:700;'>AI SpillGuard</div>
        <div style='color:#8899aa; font-size:0.8rem;'>v2.0 — Oil Spill Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### ⚙️ Detection Settings")
    threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Lower = more sensitive | Higher = more precise")

    st.divider()
    st.markdown("#### 🤖 Model Details")
    st.markdown("""
    <div style='background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px; padding:15px;'>
        <div style='color:#8899aa; font-size:0.85rem; line-height:2;'>
            🧠 <b style='color:#00d4ff'>Architecture:</b> U-Net (PyTorch)<br>
            📡 <b style='color:#00d4ff'>Dataset:</b> Aerial/Drone Images<br>
            📐 <b style='color:#00d4ff'>Input Size:</b> 256×256 BGR<br>
            🎯 <b style='color:#00d4ff'>Best Threshold:</b> 0.5<br>
            📊 <b style='color:#00d4ff'>Val Dice Score:</b> 0.9189<br>
            ⚡ <b style='color:#00d4ff'>Loss:</b> BCE + Dice
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📈 Training Summary")
    st.markdown("""
    <div style='background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px; padding:15px;'>
        <div style='color:#8899aa; font-size:0.85rem; line-height:2;'>
            🖼️ <b style='color:#00d4ff'>Total Images:</b> 1,776<br>
            🏋️ <b style='color:#00d4ff'>Train:</b> 1,243<br>
            ✅ <b style='color:#00d4ff'>Val:</b> 266<br>
            🧪 <b style='color:#00d4ff'>Test:</b> 267<br>
            🔄 <b style='color:#00d4ff'>Epochs:</b> 25<br>
            ⚡ <b style='color:#00d4ff'>Optimizer:</b> Adam
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============ Hero ============
st.markdown("""
<div class='hero'>
    <div class='hero-title'>🛢️ AI SpillGuard</div>
    <div class='hero-sub'>AI-Powered Oil Spill Detection System using Aerial & Drone Imagery</div>
    <div class='hero-badges'>
        <span class='badge'>🚁 Aerial/Drone Images</span>
        <span class='badge'>🧠 U-Net Deep Learning</span>
        <span class='badge'>⚡ Real-time Detection</span>
        <span class='badge'>🌊 Marine Environment</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============ Main Layout ============
col1, col2 = st.columns([1, 1.6], gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader("", type=["jpg","png","jpeg"],
                                help="Upload an aerial or drone oil spill image")
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption=f"📁 {uploaded.name}", use_container_width=True)
        st.markdown(f"""
        <div style='background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px;
                    padding:12px; margin-top:10px;'>
            <div style='color:#8899aa; font-size:0.85rem;'>
                📏 Size: {img.size[0]}×{img.size[1]} px &nbsp;|&nbsp;
                📂 Format: {img.format or uploaded.type.split('/')[1].upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        detect_btn = st.button("🔍  Analyze — Detect Oil Spill", type="primary")
    else:
        st.markdown("""
        <div class='upload-card'>
            <div style='font-size:3rem'>🛰️</div>
            <div style='color:#00d4ff; font-weight:600; margin:10px 0;'>Drop Image Here</div>
            <div style='color:#8899aa; font-size:0.85rem;'>Supports JPG, PNG, JPEG</div>
        </div>
        """, unsafe_allow_html=True)
        detect_btn = False

with col2:
    if uploaded and detect_btn:
        with st.spinner("🔍 Analyzing image..."):
            model_loaded = load_model()
            img_bgr      = preprocess(img)        # BGR float32 — matches training
            pred         = predict(model_loaded, img_bgr)
            pred_binary  = (pred > threshold).astype(np.uint8)
            oil_pct      = pred_binary.mean() * 100
            confidence   = float(pred.max())

        if oil_pct > 5:
            st.markdown(f"""
            <div class='alert-danger'>
                <div class='alert-danger-title'>🚨 OIL SPILL DETECTED!</div>
                <div style='color:#ffaaaa; margin-top:8px;'>
                    Approximately <b>{oil_pct:.1f}%</b> of the scanned area shows oil contamination.
                    Immediate environmental response recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='alert-safe'>
                <div class='alert-safe-title'>✅ No Oil Spill Detected</div>
                <div style='color:#aaffcc; margin-top:8px;'>
                    The scanned area appears clean. Oil coverage: <b>{oil_pct:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""<div class='result-header'>📊 Detection Metrics</div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-grid'>
            <div class='metric-card'>
                <div class='metric-val'>{oil_pct:.1f}%</div>
                <div class='metric-lbl'>🛢️ Oil Coverage</div>
            </div>
            <div class='metric-card'>
                <div class='metric-val'>{confidence:.3f}</div>
                <div class='metric-lbl'>🎯 Max Confidence</div>
            </div>
            <div class='metric-card'>
                <div class='metric-val'>{threshold}</div>
                <div class='metric-lbl'>⚙️ Threshold</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div class='result-header'>🖼️ Visual Analysis</div>""", unsafe_allow_html=True)

        # Display: convert BGR back to RGB for correct colours
        img_display = bgr_to_rgb(img_bgr)

        overlay  = img_display.copy().astype(np.float32) / 255.0
        red_mask = pred_binary == 1
        overlay[red_mask, 0] = np.clip(overlay[red_mask, 0] + 0.6, 0, 1)
        overlay[red_mask, 1] = overlay[red_mask, 1] * 0.3
        overlay[red_mask, 2] = overlay[red_mask, 2] * 0.3

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.patch.set_facecolor('#0d1b2a')

        titles    = ['🖼️ Original Image', '🔥 Prediction Heatmap', '🔴 Overlay (Red=Oil)']
        plot_imgs = [img_display.astype(np.float32)/255.0, pred, overlay]
        cmaps     = [None, 'hot', None]

        for ax, title, im, cmap in zip(axes, titles, plot_imgs, cmaps):
            ax.set_facecolor('#0d1b2a')
            ax.imshow(np.clip(im, 0, 1), cmap=cmap)
            ax.set_title(title, color='#00d4ff', fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e3a5f')

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

    elif not uploaded:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0d1b2a, #1a2744);
                    border:1px solid #1e3a5f; border-radius:15px; padding:60px;
                    text-align:center; margin-top:20px;'>
            <div style='font-size:4rem; margin-bottom:15px;'>🌊</div>
            <div style='color:#00d4ff; font-size:1.3rem; font-weight:600; margin-bottom:10px;'>
                Ready to Detect Oil Spills
            </div>
            <div style='color:#8899aa; font-size:0.95rem; line-height:1.8;'>
                Upload an aerial or drone image on the left<br>
                to begin AI-powered oil spill detection
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded and not detect_btn:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0d1b2a, #1a2744);
                    border:1px solid #1e3a5f; border-radius:15px; padding:40px; text-align:center;'>
            <div style='font-size:3rem; margin-bottom:15px;'>👈</div>
            <div style='color:#00d4ff; font-size:1.2rem; font-weight:600;'>
                Click "Analyze" to detect oil spill
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============ Footer ============
st.divider()
st.markdown("""
<div style='text-align:center; color:#445566; font-size:0.85rem; padding:10px;'>
    🛢️ AI SpillGuard — Built with PyTorch & Streamlit &nbsp;|&nbsp;
    🚁 Aerial/Drone Dataset &nbsp;|&nbsp;
    🧠 U-Net Deep Learning Model &nbsp;|&nbsp;
    📊 Val Dice: 0.9189
</div>
""", unsafe_allow_html=True)