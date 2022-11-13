import copy

import torch
import torch.nn.functional as F
from transformers import GPTNeoForCausalLM, AutoTokenizer, pipeline
import numpy as np
from tqdm import trange


import streamlit as st


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        pass


MODEL_CLASSES = {
    #'EleutherAI/gpt-neo-125M': (GPTNeoForCausalLM, AutoTokenizer),
    'lcw99/gpt-neo-1.3B-ko': (GPTNeoForCausalLM, AutoTokenizer),
}


# @st.cache
def load_tokenizer(model_name):
    model_class, tokenizer_class = MODEL_CLASSES[model_name]

    #model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)

    #model.to(device)
    #model.eval()
    return tokenizer


if __name__ == "__main__":

    # Selectors
    model_name = st.sidebar.selectbox("Model", list(MODEL_CLASSES.keys()))
    length = st.sidebar.slider("Length", 100, 2048, 500)
    temperature = st.sidebar.slider("Temperature", 0.0, 3.0, 0.8)
    top_k = st.sidebar.slider("Top K", 0, 10, 0)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_name)

    # making a copy so streamlit doesn't reload models
    #model = copy.deepcopy(model)
    tokenizer = copy.deepcopy(tokenizer)

    text_generation = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
    )

    st.title("Text generation with transformers")
    raw_text = st.text_input("Enter start text and press enter")
    if raw_text:
        generated = text_generation(
            raw_text,
            max_length=length,
            do_sample=True,
            min_length=100,
            num_return_sequences=3,
            top_p=top_p,
            top_k=top_k
        )
        st.write(*generated)