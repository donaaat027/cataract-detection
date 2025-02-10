import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# ======================================
# CONSTANTS & CONFIGURATIONS
# ======================================
CLASS_LABELS = ["Katarak", "Non-Katarak"]
MODEL_INPUT_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 5
ALLOWED_FILE_TYPES = ["image/jpeg", "image/png", "image/jpg"]

def inject_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ======================================
# HEADER SECTION (Enhanced)
# ======================================
def show_header():
    with st.container():
        st.markdown('<div class="header">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            st.markdown('<div class="eye-icon">👁️</div>', unsafe_allow_html=True)
            
        with col2:
            st.title("Deteksi Katarak dengan AI")
            st.markdown("**Unggah gambar mata untuk analisis cepat menggunakan model deep learning**")
            
        with col3:
            st.empty()
            
        st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# MODEL FUNCTIONS 
# ======================================
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = r"best_model_2.keras"
    return tf.keras.models.load_model(model_path)

def process_image(image):
    img = image.resize(MODEL_INPUT_SIZE)
    img_array = np.array(img, dtype=np.float32) 
    return np.expand_dims(img_array, axis=0)

def predict(image):
    model = load_model()
    processed_img = process_image(image)
    return model.predict(processed_img)

# ======================================
# RESULT DISPLAY COMPONENT
# ======================================
def display_results(prob_cataract, Prob_no_cataract):
    result_class = "cataract-box" if prob_cataract > Prob_no_cataract else "healthy-box"
    result_icon = "⚠️" if prob_cataract > Prob_no_cataract else "✅"
    result_text = "Terindikasi Katarak" if prob_cataract > Prob_no_cataract else "Mata Sehat"
    recommendation = ("Segera konsultasi dengan dokter spesialis mata" 
                     if prob_cataract > Prob_no_cataract 
                     else "Lakukan pemeriksaan rutin 6 bulan sekali")

    with st.container():
        st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)
        
        # Result Header
        st.markdown(f"""
        <div class="prediction-text">
            {result_icon} {result_text}
        </div>
        """, unsafe_allow_html=True)

        # Probability Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilitas Katarak", 
                    f"{prob_cataract * 100:.2f}%",
                    help="Kemungkinan adanya indikasi katarak")
            st.progress(float(prob_cataract))

        with col2:
            st.metric("Probabilitas Non-Katarak", 
                    f"{Prob_no_cataract * 100:.2f}%",
                    help="Kemungkinan mata dalam kondisi non-katarak")
            st.progress(float(Prob_no_cataract))

        # Recommendation
        if prob_cataract > Prob_no_cataract:
            st.error(f"**Rekomendasi:** {recommendation}")
        else:
            st.success(f"**Rekomendasi:** {recommendation}")

        st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# SIDEBAR COMPONENTS
# ======================================
# def show_sidebar():
#     with st.sidebar:
#         with st.expander("📌 **Panduan Penggunaan**", expanded=True):
#             st.markdown("""
#             1. Pilih gambar mata yang jelas
#             2. Pastikan area mata terlihat fokus
#             3. Tunggu hasil analisis (1-5 detik)
#             4. Konsultasi dokter tetap diperlukan
#             """)

#         with st.expander("📊 **Statistik Model**"):
#             st.markdown("""
#             **Performansi Model:**
#             - Akurasi: 100%
#             - Presisi: 100%
#             - Recall: 100%
#             - F1-Score: 100%
#             """)
            
#             # Model performance visualization
#             metrics = pd.DataFrame({
#                 'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
#                 'Value': [1.00, 1.00, 1.00, 1.00]
#             })
#             st.bar_chart(metrics.set_index('Metric'))

#         with st.expander("ℹ️ **Informasi Teknis**"):
#             st.markdown("""
#             - **Arsitektur:** MobileNetV3 Large
#             - **Dataset:** 800 gambar mata
#             - **Augmentasi Data:** Zoom, Shear, Flip,
#             - **Optimizer:** Adam (lr=0.001)
#             - **Pelatihan:** 10 epoch
#             """)

#         st.markdown("---")
#         st.markdown("""
#         **Disclaimer Medis:**  
#         Hasil analisis ini bersifat informatif awal dan tidak menggantikan diagnosis medis profesional. 
#         Selalu konsultasikan dengan dokter spesialis mata untuk pemeriksaan lengkap.
#         """)

# ======================================
# MAIN APP FUNCTIONALITY
# ======================================
def app():
    show_header()
    
    main_col, info_col = st.columns([3, 1])
    
    with main_col:
        # File Upload Section
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "**UNGGAH GAMBAR MATA**",
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPEG, JPG, PNG"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                # Validasi file
                if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.error(f"❌ Ukuran file melebihi {MAX_FILE_SIZE_MB}MB")
                    return
                
                if uploaded_file.type not in ALLOWED_FILE_TYPES:
                    st.error("❌ Format file tidak didukung. Gunakan format JPEG/PNG")
                    return

                # Proses gambar
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, 
                        use_container_width=True, 
                        caption="Gambar yang Diunggah",
                        output_format="PNG")

                # Prediksi
                with st.spinner("🔄 Sedang menganalisis gambar..."):
                    start_time = time.time()
                    prediction = predict(image)
                    process_time = time.time() - start_time
                    
                    threshold = 0.55
                    prob_cataract = prediction[0][0]
                    prob_no_cataract = prediction[0][1]

                    # Perbaikan logika threshold
                    if max(prob_cataract, prob_no_cataract) < threshold:
                        st.warning("""
                        **Hasil Tidak Pasti**\n
                        Gambar tidak terdeteksi sebagai mata yang jelas.
                        Silakan unggah gambar dengan kriteria:
                        - Gambar mata terbuka lebar
                        - Area mata terlihat dengan jelas
                        - Pencahayaan cukup
                        - Tidak blur
                        """)
                    else:
                        display_results(prob_cataract, prob_no_cataract)
                    st.caption(f"⏱️ Waktu pemrosesan: {process_time:.2f} detik")

            except Exception as e:
                st.error(f"""
                **Gagal memproses gambar:**
                ```python
                {str(e)}
                ```
                Pastikan gambar memenuhi kriteria:
                1. Format JPG/PNG
                2. Ukuran < {MAX_FILE_SIZE_MB}MB
                3. Gambar mata yang jelas
                """)
                st.stop()

    # with info_col:
    #     show_sidebar()

# ======================================
# RUN THE APP
# ======================================
if __name__ == "__main__":
    app()