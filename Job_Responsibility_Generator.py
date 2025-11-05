import streamlit as st
import torch
from gpt_arcitecture import GPTModel, generate

experience_selection = [
    "Fresher",
    "Experienced",
    "Entry-Level",
    "Mid-Level",
    "Senior-Level"
]

title_selection = [
 'Python Developer',
 'AI Engineer',
 'Frontend Developer',
 'Software Developer',
 'JavaScript Developer',
 'Data Analyst',
 'Technical Writer',
 'Full Stack Developer',
 'Site Reliability Engineer',
 'Software Engineer',
 'Product Manager',
 'Test Automation Engineer',
 'Backend Developer',
 'Blockchain Developer',
 'Java Developer',
 'Business Analyst',
 'SEO Specialist',
 'Frontend Developer',
 'Game Developer',
 'Big Data Specialist',
 'Software Engineer',
 'Cloud Engineer',
 'Solutions Architect',
 'Software Tester',
 'Fintech Engineer',
 'Software Developer',
 'AI Prompt Engineer',
 'Full Stack Developer',
 'Content Writer',
 'Operations Manager',
 'Web Developer',
 'Backend Developer',
 'UI Designer',
 'Copywriter',
 'AR/VR Developer',
 'Ethical Hacker',
 'Market Research Analyst',
 'BI Analyst',
 'QA Engineer',
 'Product Designer',
 '.NET Developer',
 'Robotics Engineer',
 'QA Engineer',
 'Graphic Designer',
 'IoT Engineer',
 'Marketing Specialist',
 'Project Manager',
 'Digital Marketing Specialist',
 'Vibe Coder',
 'Cloud Engineer',
 'Sales Executive',
 'Data Engineer',
 'BI Analyst',
 'DevOps Engineer',
 'AI Engineer',
 'Test Automation Engineer',
 'iOS Developer'
]

CONFIG = {
    "n_vocab": 50257,    # Vocabulary size
    "n_ctx": 256, # Context length
    "n_embd": 768,         # Embedding dimension
    "n_head": 12,          # Number of attention heads
    "n_layer": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}


st.title("Job Responsibilities Generator")
st.write("---")

col1, col2 = st.columns(2)

experience_level = col1.selectbox("Experience Level", options=experience_selection)
job_title = col1.selectbox("Job Title", options=title_selection)


temp = col2.select_slider("Temperature", options=[0.5,0.8,1,1.2,1.4,2],value=1)
top_k = col2.number_input("Top K", min_value = 1, max_value=100, value=None)

start = st.text_input("Direction",value="", placeholder="Implement...")

st.write("  ")
submit = st.button("Generate", use_container_width=True)

if submit:
    query = job_title + "-" + experience_level+" : " +start
    
    device = torch.device("cpu")
    model = GPTModel(CONFIG).to(device)
    model.load_state_dict(torch.load("./archive/job_model.pth"), device)

    output = generate(model, query, max_new_tokens=60, context_size=256, temperature=temp, top_k=top_k, device = torch.device("cpu"))
    output = output.split(":",1)[-1]
    st.write("\n - " + "\n - ".join(output.split(";")))