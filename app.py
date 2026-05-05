import streamlit as st
import os, torch, cv2, numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io
import base64

st.set_page_config(page_title="Lighting Inc. Studio", layout="wide")
st.title("💡 Lighting Inc. | Virtual Studio")

@st.cache_resource
def load_model():
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cpu")
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return midas, midas_transforms

def get_natural_kelvin(warmth_val):
    if warmth_val > 0.5:
        r, g, b = 255, 255 - (110 * (warmth_val - 0.5) * 2), 255 - (220 * (warmth_val - 0.5) * 2)
    else:
        r, g, b = 255 - (180 * (0.5 - warmth_val) * 2), 255 - (60 * (0.5 - warmth_val) * 2), 255
    return np.array([b, g, r], dtype=np.float32) / 255.0

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 1. Upload Assets")
    uploaded_room = st.file_uploader("Upload Room (JPG)", type=["jpg", "jpeg"])
    uploaded_lamp = st.file_uploader("Upload Light (PNG)", type=["png"])

    st.divider()
    canvas_result = None
    
    if uploaded_room:
        st.header("🧹 2. Object Cleanup")
        st.markdown("Paint over old fixtures:")
        
        pil_room_src = Image.open(uploaded_room).convert("RGB")
        
        # Convert to Base64 to force the image to show
        buffered = io.BytesIO()
        pil_room_src.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        bg_data = f"data:image/png;base64,{img_str}"
        
        w_src, h_src = pil_room_src.size
        c_height = 300
        c_width = int((c_height / h_src) * w_src)

        # --- THE BYPASS: Injecting the image via CSS ---
        st.markdown(
            f'<style>div[data-testid="stCanvas"] {{ background-image: url({bg_data}); background-size: contain; background-repeat: no-repeat; }}</style>',
            unsafe_allow_html=True
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1.0)",
            stroke_width=20,
            stroke_color="rgba(255, 255, 255, 1.0)",
            background_color="rgba(0,0,0,0)", # Transparent so CSS image shows through
            update_streamlit=True,
            height=c_height,
            width=c_width,
            drawing_mode="freedraw",
            key="cleanup_canvas"
        )
    
    st.divider()
    st.header("🎛️ 3. Staging Controls")
    bright = st.slider("Intensity", 0, 255, 130)
    warmth = st.slider("Warmth", 0.0, 1.0, 0.5)
    scale = st.slider("Scale", 0.05, 0.8, 0.2)
    x_pos = st.slider("X Position", 0, 1000, 500)
    y_pos = st.slider("Y Position", 0, 1000, 400)

# --- MAIN ENGINE ---
if uploaded_room and uploaded_lamp:
    # 1. Initialize Session State for Depth
    if "depth_map" not in st.session_state or st.session_state.get("last_room") != uploaded_room.name:
        st.session_state.last_room = uploaded_room.name
        st.session_state.depth_map = None

    # 2. Image Prep
    room_bytes = uploaded_room.getvalue()
    room_img = cv2.imdecode(np.frombuffer(room_bytes, np.uint8), cv2.IMREAD_COLOR)
    lamp_img = cv2.imdecode(np.frombuffer(uploaded_lamp.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    h, w = room_img.shape[:2]
    ax, ay = int((x_pos/1000)*w), int((y_pos/1000)*h)

    # 3. Cleanup Old Fixture
    if canvas_result is not None and canvas_result.image_data is not None:
        mask = canvas_result.image_data[:, :, 3] 
        if np.any(mask > 0):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            room_img[mask_resized > 0] = [255, 255, 255] 

    # 4. Optimized Depth Analysis
    if st.session_state.depth_map is None:
        with st.spinner("🚀 Turbo-charging 3D Map..."):
            midas, transform = load_model()
            # Downscale for speed
            small_room = cv2.resize(np.array(pil_room_src), (w//2, h//2))
            input_batch = transform(small_room).to("cpu")
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze()
            st.session_state.depth_map = prediction.cpu().numpy()
    
    depth_map = st.session_state.depth_map

    # 5. Lighting Logic (Fast Math)
    color_tint = get_natural_kelvin(warmth)
    target_d = depth_map[min(ay, h-1), min(ax, w-1)]
    glow = np.zeros((h, w), dtype=np.float32)
    cv2.circle(glow, (ax, ay), 350, 1, -1)
    glow = cv2.GaussianBlur(glow, (151, 151), 0)
    
    # Quick lighting calculation
    depth_inf = np.clip(1.0 - (np.abs(depth_map - target_d) / 50.0), 0, 1)
    final_mask = (glow * depth_inf * bright)
    tinted_glow = cv2.merge([final_mask * color_tint[0], final_mask * color_tint[1], final_mask * color_tint[2]])
    room_lit = np.clip(room_img.astype(np.float32) + tinted_glow, 0, 255).astype(np.uint8)

    # 6. Product Overlay
    hl, wl = int(lamp_img.shape[0]*scale), int(lamp_img.shape[1]*scale)
    if hl > 0 and wl > 0:
        lamp_r = cv2.resize(lamp_img, (wl, hl))
        y1, y2, x1, x2 = max(0, ay-hl//2), min(h, ay+hl//2), max(0, ax-wl//2), min(w, ax+wl//2)
        ls = lamp_r[0:(y2-y1), 0:(x2-x1)]
        if ls.shape[2] == 4:
            alpha = cv2.merge([ls[:,:,3]/255.0]*3)
            room_lit[y1:y2, x1:x2] = (ls[:,:,:3] * alpha + room_lit[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)

    st.image(cv2.cvtColor(room_lit, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # 7. Export
    res_pil = Image.fromarray(cv2.cvtColor(room_lit, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    res_pil.save(buf, format="PNG")
    st.download_button("📸 Download Staged Photo", buf.getvalue(), f"Staged_{uploaded_room.name}", "image/png")

    # Lighting
    color_tint = get_natural_kelvin(warmth)
    target_d = depth_map[min(ay, h-1), min(ax, w-1)]
    glow = np.zeros((h, w), dtype=np.float32)
    cv2.circle(glow, (ax, ay), 350, 1, -1)
    glow = cv2.GaussianBlur(glow, (151, 151), 0)
    depth_inf = np.clip(1.0 - (np.abs(depth_map.astype(np.float32) - target_d) / 50.0), 0, 1)
    final_mask = (glow * depth_inf * bright)
    tinted_glow = cv2.merge([final_mask * color_tint[0], final_mask * color_tint[1], final_mask * color_tint[2]])
    room_lit = np.clip(room_img.astype(np.float32) + tinted_glow, 0, 255).astype(np.uint8)

    # Overlay
    hl, wl = int(lamp_img.shape[0]*scale), int(lamp_img.shape[1]*scale)
    if hl > 0 and wl > 0:
        lamp_r = cv2.resize(lamp_img, (wl, hl))
        y1, y2, x1, x2 = max(0, ay-hl//2), min(h, ay+hl//2), max(0, ax-wl//2), min(w, ax+wl//2)
        ls = lamp_r[0:(y2-y1), 0:(x2-x1)]
        if ls.shape[2] == 4:
            alpha = cv2.merge([ls[:,:,3]/255.0]*3)
            room_lit[y1:y2, x1:x2] = (ls[:,:,:3] * alpha + room_lit[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)

    st.image(cv2.cvtColor(room_lit, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Export
    res_pil = Image.fromarray(cv2.cvtColor(room_lit, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    res_pil.save(buf, format="PNG")
    st.download_button("📸 Download Staged Photo", buf.getvalue(), f"Staged_{uploaded_room.name}", "image/png")
else:
    st.info("Upload room and product photos in the sidebar.")
