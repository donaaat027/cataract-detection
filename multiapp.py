import streamlit as st
import pandas as pd

def show_sidebar():
    with st.sidebar:
        # Navigation styling
        st.markdown("""
        <style>
            div[role="radiogroup"] > label > div:first-child {
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
                transition: all 0.3s;
            }
            div[role="radiogroup"] > label > div:first-child:hover {
                background: #f0f2f6;
            }
            .sidebar .sidebar-content {
                padding: 4rem 1rem !important;
            }
        </style>
        """, unsafe_allow_html=True)

        with st.expander("📌 **Panduan Penggunaan**", expanded=True):
            st.markdown("""
            1. 🖼️ Pilih gambar mata yang jelas
            2. 🔍 Pastikan area mata terlihat fokus
            3. ⏳ Tunggu hasil analisis (1-5 detik)
            4. 🩺 Konsultasi dokter tetap diperlukan
            """)

        with st.expander("📊 **Statistik Model**"):
            cols = st.columns(2)
            with cols[0]:
                st.metric("Akurasi", "100%")
                st.metric("Presisi", "100%")
            with cols[1]:
                st.metric("Recall", "100%")
                st.metric("F1-Score", "100%")

        with st.expander("ℹ️ **Informasi Teknis**"):
            st.markdown("""
            - **🧠 Arsitektur:** MobileNetV3 Large
            - **📁 Dataset:** 800 gambar mata
            - **🔄 Augmentasi Data:** Zoom, Shear, Flip
            - **⚙️ Optimizer:** Adam (lr=0.001)
            - **⏱️ Pelatihan:** 10 epoch
            """)

        st.markdown("---")
        st.warning("""
        **Disclaimer Medis:**  
        Hasil analisis ini bersifat informatif awal dan tidak menggantikan diagnosis medis profesional. 
        Selalu konsultasikan dengan dokter spesialis mata untuk pemeriksaan lengkap.
        """)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # Render navigation dengan styling improved
        st.sidebar.markdown("## 🧭 Navigasi Aplikasi")
        app = st.sidebar.radio(
            '',
            self.apps,
            format_func=lambda app: f"👉 {app['title']}",
            label_visibility="collapsed"
        )
        
        # Render sidebar content
        show_sidebar()
        
        # Eksekusi app function
        app['function']()