import streamlit as st
def run():
    st.title("Upload debug")
    f = st.file_uploader("Upload anything", type=None, accept_multiple_files=False)
    if f:
        st.write({"name": f.name, "size": len(f.getbuffer())})
