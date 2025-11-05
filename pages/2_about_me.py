import streamlit as st

st.title("About Me")



st.space()
st.space()

st.write("This was just a project resulting from me wanting to try out how easy streamlit really is and learning transformers/ llm's from scratch, so why not try to implement is on some dataset i find cool")

st.space()

st.write("Well thats it, following is my portfolio, another ML project, and my linkedin if anyone's interested")


st.space()
st.space()

c1,c2,c3 = st.columns(3)

c1.link_button("Portfolio","https://ashahlawat.dev/intro")
c2.link_button("Linkedin", "https://www.linkedin.com/in/ashwani-ahlawat/")
c3.link_button("ModeLsmith", "https://modelsmith.app")