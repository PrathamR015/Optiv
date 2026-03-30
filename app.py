import streamlit as st
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# -------------------------
# 🚀 LOAD MODEL (CACHED)
# -------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "iiiorg/piiranha-v1-detect-personal-information"
    )
    model = AutoModelForTokenClassification.from_pretrained(
        "iiiorg/piiranha-v1-detect-personal-information"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


tokenizer, model, device = load_model()

# -------------------------
# 🔍 REGEX PATTERNS
# -------------------------
PHONE_REGEX = r'(\+?\d{1,3}[-\s]?)?\d{10}'
EMAIL_REGEX = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
DATE_REGEX = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'


def mask_with_regex(text):
    text = re.sub(PHONE_REGEX, "[PHONE]", text)
    text = re.sub(EMAIL_REGEX, "[EMAIL]", text)
    text = re.sub(DATE_REGEX, "[DATE]", text)
    return text


# -------------------------
# 🤖 MODEL MASKING
# -------------------------
def model_mask(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    encoded_inputs = tokenizer.encode_plus(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        max_length=512,
        truncation=True
    )

    offset_mapping = encoded_inputs['offset_mapping']
    tokens = tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'])

    out_chars = []
    last_idx = 0
    prev_label = 'O'
    current_label = None

    for span, token, pred_label_idx in zip(offset_mapping, tokens, predictions):
        start, end = span

        if start == end:
            continue

        label = model.config.id2label[pred_label_idx]

        if label.startswith("I-"):
            label = label.replace("I-", "")

        if label != 'O':
            if prev_label == 'O':
                if last_idx < start:
                    out_chars.append(text[last_idx:start])
                current_label = label
        else:
            if prev_label != 'O':
                out_chars.append(f' [{current_label}] ')
                last_idx = end

        prev_label = label

    if prev_label != 'O':
        out_chars.append(f' [{current_label}] ')
        last_idx = end

    if last_idx < len(text):
        out_chars.append(text[last_idx:])

    return ' '.join(out_chars).replace("  ", " ").strip()


# -------------------------
# 🔐 FINAL MASK FUNCTION
# -------------------------
def mask_pii(text, aggregate=False):
    # Step 1: Regex masking (phones, email, date)
    text = mask_with_regex(text)

    if aggregate:
        return "[REDACTED TEXT]"

    # Step 2: Model masking (names, locations, etc.)
    text = model_mask(text)

    return text


# -------------------------
# 🎨 STREAMLIT UI
# -------------------------
st.set_page_config(page_title="PII Masking App", layout="centered")

st.title("🔐 PII Masking App")
st.markdown("Mask personal information using AI + Regex")

# Input
text_input = st.text_area("✍️ Enter your text:", height=200)

# Mode selection
mode = st.radio(
    "Select Mode:",
    ["Detailed Masking", "Full Redaction"]
)

# Button
if st.button("🚀 Mask PII"):
    if text_input.strip():

        if mode == "Full Redaction":
            result = mask_pii(text_input, aggregate=True)
        else:
            result = mask_pii(text_input, aggregate=False)

        st.subheader("✅ Output")
        st.code(result)

    else:
        st.warning("⚠️ Please enter some text.")


# -------------------------
# 💡 EXAMPLE
# -------------------------
with st.expander("💡 Example Input"):
    st.write(
        "My name is Pratham, phone number is +9190803470, email is pratham@gmail.com, born on 01/01/2000"
    )