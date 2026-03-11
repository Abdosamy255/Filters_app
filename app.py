

import io
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


PAGE_TITLE = "Image Filter Studio"
FILTER_OPTIONS = ("None", "Black and White", "Brightness", "Style", "Vintage", "HDR")

st.set_page_config(page_title=PAGE_TITLE, layout="wide")


def _inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1.2rem;
                padding-bottom: 1.2rem;
            }
            h1 {
                margin-bottom: 0.2rem;
            }
            [data-testid="stHorizontalBlock"] > div {
                align-self: stretch;
            }
            [data-testid="stForm"] {
                border: 1px solid #d9e2ec;
                border-radius: 12px;
                padding: 0.85rem 0.95rem;
                background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%);
            }
            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 12px;
            }
            div.stButton > button,
            div.stDownloadButton > button,
            div[data-testid="stFormSubmitButton"] > button {
                width: 100%;
                min-height: 42px;
                border-radius: 10px;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _validate_image(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("No image provided")
    if img.size == 0:
        raise ValueError("Empty image provided")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim != 3:
        raise ValueError("Unsupported image format")
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.shape[2] != 3:
        raise ValueError("Only RGB/RGBA images are supported")
    return img


def black_and_white(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_validate_image(img), cv2.COLOR_RGB2GRAY)


def adjust_brightness(img: np.ndarray, level: int) -> np.ndarray:
    return cv2.convertScaleAbs(_validate_image(img), alpha=1.0, beta=int(level))


def style_filter(img: np.ndarray, sigma_s: int, sigma_r: float) -> np.ndarray:
    bgr = _to_bgr(_validate_image(img))
    stylized = cv2.stylization(bgr, sigma_s=int(sigma_s), sigma_r=float(sigma_r))
    return _to_rgb(stylized)


def vintage_filter(img: np.ndarray, level: int) -> np.ndarray:
    src = _validate_image(img)
    h, w = src.shape[:2]
    level = max(1, int(level))
    sigma_x = max(1.0, w / float(level * 2.0))
    sigma_y = max(1.0, h / float(level * 2.0))
    kx = cv2.getGaussianKernel(w, sigma_x)
    ky = cv2.getGaussianKernel(h, sigma_y)
    mask = (ky * kx.T) / (ky * kx.T).max()
    out = src.astype(np.float32).copy()
    for c in range(3):
        out[:, :, c] *= mask
    return np.clip(out, 0, 255).astype(np.uint8)


def hdr_filter(img: np.ndarray, level: int, sigma_s: int, sigma_r: float) -> np.ndarray:
    src_bgr = _to_bgr(_validate_image(img))
    bright = cv2.convertScaleAbs(src_bgr, alpha=1.0, beta=int(level))
    enhanced = cv2.detailEnhance(bright, sigma_s=int(sigma_s), sigma_r=float(sigma_r))
    return _to_rgb(enhanced)


def _to_png_bytes(np_img: np.ndarray) -> Tuple[bytes, str]:
    pil = Image.fromarray(np_img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _vertical_gradient(height: int, width: int, top_rgb: Tuple[int, int, int], bottom_rgb: Tuple[int, int, int]) -> np.ndarray:
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        gradient[:, :, c] = np.linspace(top_rgb[c], bottom_rgb[c], height, dtype=np.uint8)[:, None]
    return gradient


def _build_landscape_sample() -> np.ndarray:
    sample = _vertical_gradient(360, 640, (115, 175, 255), (240, 250, 255))

    cv2.circle(sample, (520, 80), 48, (255, 232, 172), -1)
    cv2.circle(sample, (520, 80), 56, (255, 245, 210), 4)

    mountain_back = np.array([[40, 240], [200, 90], [360, 240]], dtype=np.int32)
    mountain_front = np.array([[230, 250], [420, 70], [620, 250]], dtype=np.int32)
    cv2.fillConvexPoly(sample, mountain_back, (110, 133, 165))
    cv2.fillConvexPoly(sample, mountain_front, (90, 113, 145))

    ground = np.array([[0, 220], [140, 200], [320, 220], [520, 205], [640, 220], [640, 360], [0, 360]], dtype=np.int32)
    cv2.fillPoly(sample, [ground], (74, 148, 88))

    river = np.array([[0, 300], [120, 280], [260, 290], [420, 270], [640, 300], [640, 360], [0, 360]], dtype=np.int32)
    cv2.fillPoly(sample, [river], (70, 166, 215))

    cv2.ellipse(sample, (140, 230), (70, 24), 0, 0, 360, (48, 114, 64), -1)
    cv2.ellipse(sample, (220, 250), (90, 30), 0, 0, 360, (52, 123, 69), -1)
    cv2.ellipse(sample, (340, 225), (80, 26), 0, 0, 360, (43, 106, 58), -1)

    # Small anti-aliased birds for a softer, finished look.
    cv2.ellipse(sample, (180, 84), (12, 6), 12, 200, 340, (66, 78, 100), 2, lineType=cv2.LINE_AA)
    cv2.ellipse(sample, (204, 84), (12, 6), -12, 200, 340, (66, 78, 100), 2, lineType=cv2.LINE_AA)
    return sample


def _build_portrait_sample() -> np.ndarray:
    sample = _vertical_gradient(640, 360, (246, 236, 226), (225, 205, 188))

    cv2.ellipse(sample, (180, 470), (130, 150), 0, 0, 360, (73, 99, 148), -1)
    cv2.ellipse(sample, (180, 455), (135, 120), 0, 0, 360, (87, 116, 168), -1)

    cv2.ellipse(sample, (180, 250), (86, 106), 0, 0, 360, (237, 201, 170), -1)
    cv2.ellipse(sample, (180, 188), (94, 98), 0, 200, 340, (75, 53, 47), 28, lineType=cv2.LINE_AA)
    cv2.ellipse(sample, (148, 252), (8, 5), 0, 0, 360, (90, 66, 58), -1)
    cv2.ellipse(sample, (212, 252), (8, 5), 0, 0, 360, (90, 66, 58), -1)
    cv2.ellipse(sample, (180, 285), (24, 12), 0, 18, 162, (145, 96, 84), 2, lineType=cv2.LINE_AA)

    cv2.rectangle(sample, (150, 328), (210, 376), (226, 190, 160), -1)
    cv2.ellipse(sample, (180, 410), (52, 24), 0, 0, 360, (224, 176, 146), -1)

    cv2.circle(sample, (55, 95), 22, (255, 246, 232), -1)
    cv2.circle(sample, (305, 120), 16, (255, 246, 232), -1)
    return sample


def _load_image(uploaded_file, selected_sample: str) -> np.ndarray:
    if selected_sample == "Landscape":
        return _to_rgb(_build_landscape_sample())
    if selected_sample == "Portrait":
        return _to_rgb(_build_portrait_sample())
    if uploaded_file is None:
        raise ValueError("Upload an image or choose a sample to continue")

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image") from exc

    return np.array(image)


def _read_filter_parameters(filter_name: str) -> Dict[str, float]:
    params: Dict[str, float] = {}
    if filter_name == "Brightness":
        params["level"] = st.slider("Brightness", -150, 150, 10, step=1)
    elif filter_name == "Style":
        params["sigma_s"] = st.slider("Sigma S", 0, 200, 60, step=5)
        params["sigma_r"] = st.slider("Sigma R", 0.0, 1.0, 0.45, step=0.05)
    elif filter_name == "Vintage":
        params["level"] = st.slider("Vintage Level", 1, 12, 3, step=1)
    elif filter_name == "HDR":
        params["level"] = st.slider("Brightness", -150, 150, 10, step=1)
        params["sigma_s"] = st.slider("Sigma S", 1, 200, 60, step=5)
        params["sigma_r"] = st.slider("Sigma R", 0.0, 1.0, 0.45, step=0.05)
    return params


def _apply_filter(img: np.ndarray, filter_name: str, params: Dict[str, float]) -> Tuple[np.ndarray, str]:
    if filter_name == "None":
        return img, "RGB"
    if filter_name == "Black and White":
        return black_and_white(img), "GRAY"
    if filter_name == "Brightness":
        return adjust_brightness(img, int(params.get("level", 10))), "RGB"
    if filter_name == "Style":
        return style_filter(img, int(params.get("sigma_s", 60)), float(params.get("sigma_r", 0.45))), "RGB"
    if filter_name == "Vintage":
        return vintage_filter(img, int(params.get("level", 3))), "RGB"
    if filter_name == "HDR":
        return hdr_filter(
            img,
            int(params.get("level", 10)),
            int(params.get("sigma_s", 60)),
            float(params.get("sigma_r", 0.45)),
        ), "RGB"
    raise ValueError("Unsupported filter")


def _init_state(img: np.ndarray) -> None:
    if "output_image" not in st.session_state:
        st.session_state.output_image = img
    if "output_mode" not in st.session_state:
        st.session_state.output_mode = "RGB"


def main() -> None:
    _inject_ui_styles()
    st.title(PAGE_TITLE)
    st.caption("A polished Streamlit app for applying OpenCV-based image filters.")

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        selected_sample = st.radio("Or use sample", ["None", "Landscape", "Portrait"], index=0)
        st.markdown("---")
        st.caption("Tip: images under 2MB provide smoother interaction.")

    try:
        img = _validate_image(_load_image(uploaded_file, selected_sample))
    except ValueError as exc:
        st.info(str(exc))
        st.stop()

    _init_state(img)

    col_orig, col_ctrl, col_out = st.columns([1.05, 0.9, 1.05], gap="large")

    with col_orig:
        with st.container(border=True):
            st.subheader("Original")
            st.image(img, channels="RGB", use_container_width=True)
            st.caption(f"Resolution: {img.shape[1]} x {img.shape[0]} px")

    with col_ctrl:
        st.subheader("Controls")
        with st.form("filter_form", clear_on_submit=False):
            filter_name = st.selectbox("Filter", FILTER_OPTIONS)
            params = _read_filter_parameters(filter_name)
            apply = st.form_submit_button("Apply Filter", type="primary", use_container_width=True)

        reset = st.button("Reset Result", use_container_width=True)
        if reset:
            st.session_state.output_image = img
            st.session_state.output_mode = "RGB"

    with col_out:
        with st.container(border=True):
            st.subheader("Result")
            if filter_name != "None" and not apply:
                st.info("Adjust settings, then click Apply Filter.")

            if filter_name == "None":
                st.session_state.output_image = img
                st.session_state.output_mode = "RGB"

            if apply:
                try:
                    with st.spinner("Applying filter..."):
                        output, color_mode = _apply_filter(img, filter_name, params)
                    st.session_state.output_image = output
                    st.session_state.output_mode = color_mode
                except Exception as exc:  # defensive guard for OpenCV runtime issues
                    st.error(f"Processing failed: {exc}")

            st.image(st.session_state.output_image, channels=st.session_state.output_mode, use_container_width=True)
            data, mime = _to_png_bytes(st.session_state.output_image)
            st.download_button("Download PNG", data=data, file_name="filtered.png", mime=mime, use_container_width=True)

    st.markdown("---")
    st.caption("Engineered for clean structure, safer input handling, and maintainability.")


if __name__ == "__main__":

    main()
