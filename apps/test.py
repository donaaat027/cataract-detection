# apps/model_info.py
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import plotly.express as px

def app():
    st.title("📊 Model Information Hub")
    
    # Section 1: Model Architecture
    with st.container():
        st.header("🧠 Model Architecture", divider="rainbow")
        
        # Load model
        @st.cache_resource
        def load_model():
            return tf.keras.models.load_model(r"best_model_2.keras")
        
        model = load_model()
        
        # Three columns layout
        col1, col2 = st.columns([2, 2])
        
        with col1:
            st.subheader("📋 Summary Table")
            with st.expander("📜 Expand Model Summary", expanded=True):
                summary_list = []
                model.summary(print_fn=lambda x: summary_list.append(x))
                summary_text = "\n".join(summary_list)
                st.code(summary_text, language="text")
        
        with col2:
            st.subheader("🖼️ Architecture Diagram")
            try:
                img_path = r'plot_model.png'
                
                with st.expander("📂 Lihat & Unduh Diagram Model", expanded=False):
                    model_img = Image.open(img_path)
                    st.image(model_img, use_container_width=True, caption='MobileNetV3 Large Architecture')

                    # Tombol download untuk gambar
                    with open(img_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Diagram",
                            data=file,
                            file_name="mobilenetv3_architecture.png",
                            mime="image/png"
                        )
            except Exception as e:
                st.error(f"🚨 Error loading diagram: {e}")
    st.divider()
    
    # New Section: MobileNetV3 Technical Details
    with st.container():
        st.header("📚 MobileNetV3 Large Deep Dive", divider="blue")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🧬 Architecture", "⚙️ Components", "📈 Comparison", "🚀 MobileNetV3 Highlights"])
        
        with tab1:
            col1, col2 = st.columns([3, 5])
            with col1:
                st.markdown("""
                **Block Architecture:**
                ```python
                Input (224x224x3)
                ↓
                Initial Conv (16 channels)
                ↓
                Bottleneck Blocks (15 stages):
                - Expansion Conv
                - Depthwise Conv
                - Squeeze-Excite
                - Projection Conv
                ↓
                Final Conv (960 channels)
                ↓
                Global Pooling
                ↓
                Classification Head
                ```
                """)
            with col2:
                st.image(r"main blocks mNV3.jpg",
                        caption="MobileNetV3 Block Structure",
                        use_container_width=True)
        
        with tab2:
            st.markdown("""
            **Key Components:**
            - 🔄 **Inverted Residuals**: 
              ```python
              Input → Expansion → Depthwise → SE → Projection
              ```
            - ⚡ **h-swish Activation**:
              ```python
              x * relu6(x + 3) / 6
              ```
            - 🎯 **Squeeze-Excite**:
              ```python
              GlobalAvgPool → FC → ReLU → FC → h-sigmoid → Channel Scale
              ```
            - 🏎️ **Efficient Last Stage**:
              Reduced channel count in final SE layer
            """)
        
        with tab3:
            st.markdown("""
            | Metric          | MobileNetV3-L | MobileNetV2 | EfficientNet-B0 |
            |-----------------|---------------|-------------|------------------|
            | Top-1 Accuracy  | 75.2%         | 72.0%       | 77.1%            |
            | Params (M)      | 7.0           | 3.4         | 5.3              |
            | MAdds (B)       | 0.155         | 0.300       | 0.390            |
            | CPU Latency (ms)| 51            | 76          | 98               |
            """)
            st.caption("*Benchmarks on Pixel 1 CPU (ImageNet-1K)*")
        with tab4 :
            st.markdown("""
                        **Core Innovations:**
            - 🔄 Network Architecture Search (NAS)
            - ⚡ Hardware-Aware Activation (h-swish/h-sigmoid)
            - 🎯 Squeeze-and-Excitation Optimization
            - 🏎️ Lite Reduced Squeeze-Excite
            
            **Performance:**
            - 📏 Input Resolution: 224x224
            - 🧮 7M Parameters
            - ⏱️ 155M Multiply-Adds
            - 🏆 75.2% ImageNet Top-1 Accuracy""")
    st.divider()
    
    # Section 2: Training History 
    st.header("📈 Training Progress", divider='rainbow')
    
    # Training metrics data
    epochs_data = {
        "Epoch": range(1, 11),
        "Training Accuracy": [0.7903, 0.9471, 0.9714, 0.9671, 0.9773, 0.9812, 0.9954, 0.9953, 0.9680, 0.9886],
        "Training Loss": [0.4375, 0.1706, 0.0860, 0.0946, 0.0577, 0.0504, 0.0310, 0.0249, 0.0682, 0.0419],
        "Validation Accuracy": [0.9333, 0.9733, 0.9933, 0.9933, 0.9800, 0.9867, 1.0000, 0.9933, 0.9867, 0.9867],
        "Validation Loss": [0.1473, 0.0679, 0.0406, 0.0396, 0.0654, 0.0427, 0.0154, 0.0532, 0.0709, 0.0361]
    }
    
    # Interactive training curves
    tab1, tab2 = st.tabs(["📉 Interactive Charts", "📋 Data Table"])
    
    with tab1:
        fig = px.line(
            pd.DataFrame(epochs_data),
            x="Epoch",
            y=["Training Accuracy", "Validation Accuracy"],
            title="Accuracy Evolution",
            markers=True,
            labels={"value": "Accuracy", "variable": "Dataset"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.line(
            pd.DataFrame(epochs_data),
            x="Epoch",
            y=["Training Loss", "Validation Loss"],
            title="Loss Evolution",
            markers=True,
            labels={"value": "Loss", "variable": "Dataset"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.dataframe(
            pd.DataFrame(epochs_data).style.format({
                "Training Accuracy": "{:.2%}",
                "Validation Accuracy": "{:.2%}",
                "Training Loss": "{:.4f}",
                "Validation Loss": "{:.4f}"
            }),
            use_container_width=True
        )
    
    # --- Section 3: Performance Metrics ---
    st.header("🏆 Model Performance", divider='rainbow')
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Accuracy", "99.12%", delta="+0.82% vs validation")
        st.progress(0.99)
    with col2:
        st.metric("Validation Accuracy", "99.33%", delta="-0.67% vs test")
        st.progress(0.99)
    with col3:
        st.metric("Test Accuracy", "100%", delta_color="off")
        st.progress(1.00)
    with col4:
        st.metric("F1 Score", "1.00", "Perfect Score")
        st.progress(1.00)
    
    # Detailed metrics grid
    st.subheader("📊 Detailed Performance Metrics")
    metrics_data = {
        "Dataset": ["Training", "Validation", "Test"],
        "Accuracy (%)": [99.12, 99.33, 100.0],
        "Loss": [0.0255, 0.0199, 0.0070],
        "Precision": [1.000, 0.993, 1.000],
        "Recall": [1.000, 0.993, 1.000]
    }
    
    st.dataframe(
        pd.DataFrame(metrics_data).style.format({
            "Accuracy (%)": "{:.2f}%",
            "Loss": "{:.4f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}"
        }).highlight_max(color='#90EE90').highlight_min(color='#FFCCCB', subset=["Loss"]),
        use_container_width=True
    )
    
    # Section 4: Technical Specifications (Enhanced)
    with st.container():
        st.header("⚙️ Technical Specifications", divider="orange")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("🧬 Model Configuration Code")
            with st.expander("🔍 View Architecture Code", expanded=True):
                st.code("""def create_model(use_dropout, include_dense):
    inputs = Input(shape=(224, 224, 3))
    base_model = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs,
        pooling='avg'
    )
    base_model.trainable = False
    x = base_model.output
    
    if use_dropout:
        x = Dropout(0.25)(x)
    
    if include_dense:
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
    
    outputs = Dense(2, activation='softmax')(x)
    return Model(inputs, outputs)
""", language='python')

        with col2:
            st.subheader("🔑 Architecture Features")
            st.markdown("""
            - **Base Network**: MobileNetV3-Large
            - **Optimization**:
              ```python
              optimizer = Adam(lr=0.001)
              loss = CategoricalCrossentropy()
              ```
            - **Regularization**:
              ```python
              Dropout(0.25) if enabled
              ```
            - **Augmentations**:
              ```python
              Zoom(0.2), Shear(0.2), Flip(horizontal)
              ```
            """)
            
            st.download_button(
                label="📥 Download Config",
                data=open("model_config.json").read(),
                file_name="model_config.json",
                mime="application/json"
            )

    st.divider()
    
    # New Section: MobileNetV3 Resources
    with st.container():
        st.header("📚 Additional Resources", divider="green")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Official Resources**")
            st.page_link("https://arxiv.org/abs/1905.02244", label="📄 Original Paper")
            st.page_link("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5", 
                       label="🧩 TensorFlow Hub")
        
        with col2:
            st.markdown("**Technical Guides**")
            st.page_link("https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large",
                       label="📚 TF Documentation")
            st.page_link("https://keras.io/api/applications/mobilenet/",
                       label="📘 Keras Guide")
        
        with col3:
            st.markdown("**Implementation**")
            st.page_link("https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py",
                       label="💻 TF Model Zoo")
            st.page_link("https://github.com/keras-team/keras-applications",
                       label="🔧 Keras Implementations")

if __name__ == "__main__":
    app()