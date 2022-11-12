import copy

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        progress_bar = st.progress(0)
        for step in trange(length):
            progress_bar.progress(step / length)
            inputs = {'input_ids': generated}
            if is_xlnet:
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.)

            # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for _ in set(generated.view(-1).tolist()):
                next_token_logits[_] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: #greedy sampling:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


MODEL_CLASSES = {
    'lcw99/gpt-neo-1.3B-ko': (GPTNeoForCausalLM, AutoTokenizer),
    # 'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    # 'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    # 'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    # 'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    # 'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}


@st.cache
def load_model(model_name):
    model_class, tokenizer_class = MODEL_CLASSES[model_name]

    model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":

    # Selectors
    model_name = st.sidebar.selectbox("Model", list(MODEL_CLASSES.keys()))
    length = st.sidebar.slider("Length", 20, 100, 20)
    temperature = st.sidebar.slider("Temperature", 0.0, 3.0, 0.8)
    top_k = st.sidebar.slider("Top K", 0, 10, 0)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_name)

    # making a copy so streamlit doesn't reload models
    model = copy.deepcopy(model)
    tokenizer = copy.deepcopy(tokenizer)

    st.title("Text generation with transformers")
    raw_text = st.text_input("Enter start text and press enter")
    if raw_text:
        context_tokens = tokenizer.encode(raw_text)
        out = sample_sequence(
            model=model,
            context=context_tokens,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_k,
            device=device,
            is_xlnet=bool(model_name == "xlnet"),
        )
        out = out[0, len(context_tokens):].tolist()
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
        st.write(text)