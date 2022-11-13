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
    'lcw99/gpt-neo-1.3B-ko-fp16': (GPTNeoForCausalLM, AutoTokenizer),
    'EleutherAI/gpt-neo-125M': (GPTNeoForCausalLM, AutoTokenizer),
}


# @st.cache
def load_model(model_name):
    model_class, tokenizer_class = MODEL_CLASSES[model_name]

    model = model_class.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
        gradient_checkpointing=False,
        device_map='auto',
        #revision="float16",
        #load_in_8bit=True
    )
    tokenizer = tokenizer_class.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":


    # Selectors
    model_name = st.sidebar.selectbox("Model", list(MODEL_CLASSES.keys()))
    length = st.sidebar.slider("Length", 50, 2048, 100)
    temperature = st.sidebar.slider("Temperature", 0.0, 3.0, 0.8)
    top_k = st.sidebar.slider("Top K", 0, 10, 0)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.7)

    st.title("Text generation with GPT-neo Korean")
    raw_text = st.text_input("시작하는 문장을 입력하고 엔터를 치세요.", placeholder="골프를 잘 치고 싶다면,", 
                             key="text_input1")

    if raw_text:
        st.write(raw_text)
        with st.spinner(f'loading model({model_name}) wait...'):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, tokenizer = load_model(model_name)

            # making a copy so streamlit doesn't reload models
            # model = copy.deepcopy(model)
            # tokenizer = copy.deepcopy(tokenizer)

            if False:
                text_generation = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )

        with st.spinner(f'Generating text wait...'):        
            # generated = text_generation(
            #     raw_text,
            #     max_length=length,
            #     do_sample=True,
            #     min_length=100,
            #     num_return_sequences=3,
            #     top_p=top_p,
            #     top_k=top_k
            # )
            # st.write(*generated)
            
            encoded_input = tokenizer(raw_text, return_tensors='pt')
            output_sequences = model.generate(
                input_ids=encoded_input['input_ids'].to(device),
                attention_mask=encoded_input['attention_mask'].to(device),
                max_length=length,
                do_sample=True,
                min_length=20,
                top_p=top_p,
                top_k=top_k
            )
            generated = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            #print(generated)
            st.write(generated)
            
            