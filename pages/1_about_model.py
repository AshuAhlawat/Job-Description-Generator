import streamlit as st
import pandas as pd

dataset = pd.read_csv("./archive/job_dataset.csv")
history = pd.read_csv("./archive/job_history.csv")

st.title("About the model")
st.write("GPT2 arcitecture that was open-sourced by openAI but implemented in PyTorch, trained from scratch locally on my RTX4060 for about 11 minutes before the loss platued")

st.write("### CONFIG")
st.write({
    "Vocabulary": 50257,    # Vocabulary size
    "Context Length": 256,        # Context length
    "Embedding Dimensions": 768,         # Embedding dimension
    "Causal Attention Heads": 12,          # Number of attention heads
    "Transformer Layers": 12,         # Number of layers
    "Dropout Rate": 0.1,       # Dropout rate
    "Query-Key-Value bias": False       # Query-Key-Value bias
})

st.write("### DATA")
st.write("used the following columns to train on, from the [Job Descriptions 2025](https://www.kaggle.com/datasets/adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles) dataset")

sample = dataset[["Title", "ExperienceLevel", "Responsibilities"]].sample(10).reset_index(drop=True)

st.write(sample)

st.write("### TRAINING")
st.write("The model was fed the Job title, experience level concataneted and then responsibilites and finally the <|endoftext|> token in batches of 5. So it learns to predict next word prediction of sentences starting with a Job title and experience. ")
st.line_chart(history, x="Epoch", y=["Training Loss", "Validation Loss"])
st.write("Validation loss most likely high as the dataset is small enough that the model doesn't learn to generalize well")
st.line_chart(history, x="Epoch", y="Perplexity")
st.write("Intuitively perplexity is how muh can the model narrow down the next possible word with confident. eg. 2 means, the model has to randomly guess out of 2 words it predictied")


st.write("### [CODE](https://github.com/AshuAhlawat/Job-Description-Generator)")
st.write("""
- **google cloud run** : deploy the containerized project 
- **streamlit** : used to make the web app
- **pandas** : to handle the data
- **pytorch** : training and testing
- **tiktoken** : get the bitwise pair encodings
""")