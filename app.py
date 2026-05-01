import streamlit as st
import os, torch, cv2, numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Lighting Inc. Studio", layout="wide")
st.title("💡 Lighting Inc. | Virtual Studio")

@st.cache_resource
def load_model():
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

def get_natural_kelvin(warmth_val):
    if warmth_val > 0.5:
        r, g, b = 255, 255 - (110 * (warmth_val - 0.5) * 2), 255 - (220 * (warmth_val - 0.5) * 2)
    else:
        r, g, b = 255 - (180 * (0.5 - warmth_val) * 2), 255 - (60 * (0.5 - warmth_val) * 2), 255
    return np.array([b, g, r], dtype=np.float32) / 255.0

# --- NEW: UPLOAD SECTION ---
with st.sidebar:
    st.header("📂 Project Assets")
    uploaded_room = st.file_uploader("Upload Room (JPG)", type=["jpg", "jpeg"])
    uploaded_lamp = st.file_uploader("Upload Light (PNG)", type=["png"])
    
    st.divider()
    st.header("🎛️ Controls")
    bright = st.slider("Intensity", 0, 255, 130)
    warmth = st.slider("Warmth", 0.0, 1.0, 0.5)
    scale = st.slider("Scale", 0.05, 0.8, 0.2)
    x_pos = st.slider("X Position", 0, 1000, 500)
    y_pos = st.slider("Y Position", 0, 1000, 400)

# --- MAIN ENGINE ---
if uploaded_room and uploaded_lamp:
    # Convert uploaded files to OpenCV format
    room_bytes = np.frombuffer(uploaded_room.read(), np.uint8)
    room_img = cv2.imdecode(room_bytes, cv2.IMREAD_COLOR)
    
    lamp_bytes = np.frombuffer(uploaded_lamp.read(), np.uint8)
    lamp_img = cv2.imdecode(lamp_bytes, cv2.IMREAD_UNCHANGED)
    
    h, w = room_img.shape[:2]
    ax, ay = int((x_pos/1000)*w), int((y_pos/1000)*h)

    # Depth Analysis
    d_name = f"depth_{uploaded_room.name}"
    if not os.path.exists(d_name):
        with st.spinner("Analyzing room 3D geometry..."):
            model = load_model()
            pil_room = Image.open(io.BytesIO(room_bytes)).convert("RGB")
            with torch.no_grad(): depth = model.infer_pil(pil_room)
            import matplotlib.pyplot as plt
            plt.imsave(d_name, depth, cmap="magma")
    
    depth_map = cv2.imread(d_name, cv2.IMREAD_GRAYSCALE)
    
    # Lighting Logic
    color_tint = get_natural_kelvin(warmth)
    target_d = depth_map[min(ay, h-1), min(ax, w-1)]
    glow = np.zeros((h, w), dtype=np.float32)
    cv2.circle(glow, (ax, ay), 350, 1, -1)
    glow = cv2.GaussianBlur(glow, (151, 151), 0)
    depth_inf = np.clip(1.0 - (np.abs(depth_map.astype(np.float32) - target_d) / 45.0), 0, 1)
    final_mask = (glow * depth_inf * bright)
    tinted_glow = cv2.merge([final_mask * color_tint[0], final_mask * color_tint[1], final_mask * color_tint[2]])
    room_lit = np.clip(room_img.astype(np.float32) + tinted_glow, 0, 255).astype(np.uint8)

    # Product Overlay
    hl, wl = int(lamp_img.shape[0]*scale), int(lamp_img.shape[1]*scale)
    if hl > 0 and wl > 0:
        lamp_r = cv2.resize(lamp_img, (wl, hl))
        y1, y2, x1, x2 = max(0, ay-hl//2), min(h, ay+hl//2), max(0, ax-wl//2), min(w, ax+wl//2)
        ls = lamp_r[0:(y2-y1), 0:(x2-x1)]
        alpha = cv2.merge([ls[:,:,3]/255.0]*3)
        room_lit[y1:y2, x1:x2] = (ls[:,:,:3] * alpha + room_lit[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)

    st.image(cv2.cvtColor(room_lit, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Export
    res_pil = Image.fromarray(cv2.cvtColor(room_lit, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    res_pil.save(buf, format="PNG")
    st.download_button("📸 Download Staged Photo", buf.getvalue(), f"Staged_{uploaded_room.name}", "image/png")
else:
    st.info("Please upload both a Room (JPG) and a Light (PNG) in the sidebar to begin.")    
