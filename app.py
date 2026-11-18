import skimage as ski
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import math
import streamlit as st
from PIL import Image

def imgIntensity(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def grayCmap(img_intensity):
  normalised = img_intensity/255.0
  gray = np.stack([normalised,normalised,normalised],axis = -1)
  return gray

def resize_nn(mat, max_dim=400):
    h, w = mat.shape
    scale = max_dim / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    out = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            out[i][j] = mat[min(int(i / scale), h - 1)][min(int(j / scale), w - 1)]
    return out

def rotation(matrix, theta):
  h, w = matrix.shape[:2]
  sin_theta = math.sin(theta)
  cos_theta = math.cos(theta)
  new_w = int(np.ceil(abs(w*cos_theta) + abs(h*sin_theta)))
  new_h = int(np.ceil(abs(h*cos_theta) + abs(w*sin_theta)))

  rotated_img = np.zeros((new_h, new_w) + matrix.shape[2:],dtype=matrix.dtype)
  R_matrix = np.array([[cos_theta, -sin_theta],[sin_theta, cos_theta]])

  C_old = np.array([(w-1)/2, (h-1)/2])
  C_new = np.array([(new_w-1)/2, (new_h-1)/2])

  for i in range(new_h):
      for j in range(new_w):

          p = np.array([j, i])

          p_original = C_old + R_matrix.T @ (p-C_new)

          x_old = round(p_original[0])
          y_old = round(p_original[1])

          if 0 <= x_old < w and 0 <= y_old < h:
              rotated_img[i, j] = matrix[y_old, x_old]

  return rotated_img

def scaling(matrix, s):
    h, w = matrix.shape
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    out = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            y_old = round(i / s)
            x_old = round(j / s)
            if 0 <= x_old < w and 0 <= y_old < h:
                out[i][j] = matrix[y_old][x_old]
    return out

def translation(matrix, tx, ty):
    h, w = matrix.shape
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x_old = round(j - tx)
            y_old = round(i - ty)
            if 0 <= x_old < w and 0 <= y_old < h:
                out[i][j] = matrix[y_old][x_old]
    return out

def get_input(prompt, lo, hi):
    while True:
        try:
            v = float(input(prompt))
        except ValueError:
            print("  Invalid input. Please enter a number.")
            continue
        if not (lo <= v <= hi):
            print(f"  Value must be between {lo} and {hi}.")
            continue
        return v

def show_grid(m, caption):
    fig, ax = plt.subplots(figsize=(5, 5))
    gray_img = grayCmap(m)
    ax.imshow(gray_img, vmin=0, vmax=255)
    h, w = m.shape
    ax.set_xticks(np.arange(0, w, 50))   # grid line every 50 pixels
    ax.set_yticks(np.arange(0, h, 50))
    ax.grid(True, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.tick_params(axis="both", labelsize=3) 
    ax.set_title(caption)
    st.pyplot(fig)
    plt.close(fig)

def to_display(m):
    # Streamlit expects uint8 (0-255) for display, so clip and convert
    return np.clip(m, 0, 255).astype(np.uint8)

st.set_page_config(
    page_title="Image Transformations",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Geometric Image Transformations")
st.write("Upload an image and experiment with geometric transformations.")

uploaded_file = st.file_uploader(
    "Upload your photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    rgb = np.array(Image.open(uploaded_file).convert("RGB"), dtype=float)
    original = imgIntensity(rgb)
    resized_img = resize_nn(original, max_dim=400)
    h, w = resized_img.shape

    st.subheader("Transformation Settings")

    # Selectbox is OUTSIDE the form, so changing it reruns the app immediately
    choice = st.selectbox(
        "Choose a transformation",
        ["Rotation", "Scaling", "Translation"]
    )

    with st.form("transformation_form"):

        # Parameters now change as soon as the choice changes
        if choice == "Rotation":
            angle = st.slider("Rotation angle (degrees)", -360, 360, 0)

        elif choice == "Scaling":
            s = st.slider("Scaling factor", 0.1, 5.0, 1.0, 0.1)

        elif choice == "Translation":
            tx = st.slider("Horizontal translation (pixels)", -w, w, 0)
            ty = st.slider("Vertical translation (pixels)", -h, h, 0)

        submitted = st.form_submit_button("Apply Transformation")

    if submitted:

        if choice == "Rotation":
            result = rotation(resized_img, math.radians(angle))
            caption = f"Rotation: {angle}°"

        elif choice == "Scaling":
            result = scaling(resized_img, s)
            caption = f"Scaling factor: {s}"

        elif choice == "Translation":
            result = translation(resized_img, tx, ty)
            caption = f"Translation: tx={tx}, ty={ty}"

        st.session_state["result"] = result
        st.session_state["caption"] = caption

    if "result" in st.session_state:

        st.divider()
        st.subheader("Transformation Results")

        col1, col2, _ = st.columns([1,1,1])

        with col1:
            show_grid(to_display(resized_img), "Original Grayscale Image")

        with col2:
            show_grid(to_display(st.session_state["result"]), st.session_state["caption"])

else:
    st.info("Upload a photograph to get started.")
