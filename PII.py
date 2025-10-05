import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")
model = AutoModelForTokenClassification.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# PII masking function
def mask_pii(text, aggregate_redaction=True):
    if aggregate_redaction:
        return "[redacted]"

    # Tokenize and run model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    # Get token offsets
    encoded_inputs = tokenizer.encode_plus(
        text, return_offsets_mapping=True, add_special_tokens=True
    )
    offset_mapping = encoded_inputs['offset_mapping']
    tokens = tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'])

    out_chars = []
    last_idx = 0
    prev_label = 'O'

    for span, token, pred_label_idx in zip(offset_mapping, tokens, predictions):
        start, end = span
        if start == end:
            continue  # skip special tokens

        label = model.config.id2label[pred_label_idx]
        if label != 'O':
            if prev_label == 'O':
                # Entering a PII span
                if last_idx < start:
                    out_chars.append(text[last_idx:start])
                current_label = label
            # Continue masking
        else:
            if prev_label != 'O':
                # Leaving a PII span
                out_chars.append(f'[{current_label}]')
                last_idx = end
        prev_label = label

    # If the text ends with PII
    if prev_label != 'O':
        out_chars.append(f'[{current_label}]')
        last_idx = offset_mapping[-1][1]

    if last_idx < len(text):
        out_chars.append(text[last_idx:])

    return ''.join(out_chars)


# Example usage
example_text = "My name is Dhanushkumar and I live at Chennai. My phone number is +9190803470. My email id is dkumar@gmail.com. I was born on 01/01/2000."

print("Aggregated Redaction:")
print(mask_pii(example_text, aggregate_redaction=True))

print("\nDetailed Redaction:")
print(mask_pii(example_text, aggregate_redaction=False))
