import streamlit as st
from multiapp import MultiApp
from apps import home,test 

app = MultiApp()

st.markdown("""
# Web Nata 

PENERAPAN TRANSFER LEARNING MENGGUNAKAN MOBILENETV3-LARGE UNTUK DETEKSI KATARAK PADA CITRA MATA

""")

app.add_app("Home", home.app)
app.add_app("Test", test.app)
app.run()
