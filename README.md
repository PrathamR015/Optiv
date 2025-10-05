# OPTIVE
# 🔒 PII Masking using Transformers (`iiiorg/piiranha-v1-detect-personal-information`)

This project demonstrates how to **detect and redact Personally Identifiable Information (PII)** such as names, phone numbers, email addresses, and locations using a **Transformer-based model** from Hugging Face.  
The model used — [`iiiorg/piiranha-v1-detect-personal-information`](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) — is fine-tuned for PII detection tasks.

---

## 📘 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Example Output](#example-output)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Model Information](#model-information)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## 🧠 Overview

PII (Personally Identifiable Information) refers to data that can identify a specific individual — such as name, email, phone number, address, date of birth, etc.  
This project provides a **Python script** that detects and replaces PII in text using a **Transformer model** from the Hugging Face ecosystem.

The script supports two modes:
- **Aggregated Redaction** → Masks the entire text as `[redacted]`
- **Detailed Redaction** → Replaces detected entities (e.g., names, emails) with their respective labels (e.g., `[GIVENNAME]`, `[EMAIL_ADDRESS]`)

---

## ⚙️ Key Features

✅ Detects multiple types of PII such as:
- Person names  
- Phone numbers  
- Email addresses  
- Dates of birth  
- Locations  

✅ Works on **GPU** if available for faster inference  
✅ Provides **both aggregate and detailed redaction options**  
✅ Uses a **pre-trained transformer** from Hugging Face for high accuracy  
✅ Clean, lightweight, and easy-to-extend Python script  

---

## 🧩 How It Works

1. The input text is tokenized using the model’s tokenizer.  
2. The model predicts a label (like `O`, `B-GIVENNAME`, `I-EMAIL_ADDRESS`, etc.) for each token.  
3. The script identifies spans of tokens labeled as PII.  
4. Those spans are replaced with `[ENTITY_TYPE]` (e.g., `[EMAIL_ADDRESS]`) or `[redacted]` depending on the mode.

---

## 💻 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/pii-masking-transformers.git
cd pii-masking-transformers
````

### 2️⃣ Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install torch transformers
```

*(If you plan to use CUDA, ensure PyTorch is installed with GPU support.)*

---

## 🚀 Usage

Run the script directly:

```bash
python pii_masking.py
```

You can also import and use the function in your own project:

```python
from pii_masking import mask_pii

text = "My name is Dhanushkumar and my email is dkumar@gmail.com."
print(mask_pii(text, aggregate_redaction=False))
```

---

## 🧾 Example Output

**Input:**

```
My name is Dhanushkumar and I live at Chennai. 
My phone number is +9190803470. 
My email id is dkumar@gmail.com. 
I was born on 01/01/2000.
```

**Output:**

**Aggregated Redaction:**

```
[redacted]
```

**Detailed Redaction:**

```
My name is [GIVENNAME] and I live at [LOCATION]. 
My phone number is [PHONE_NUMBER]. 
My email id is [EMAIL_ADDRESS]. 
I was born on [DATE_TIME].
```

---

## 📁 Project Structure

```
📦 pii-masking-transformers
 ┣ 📜 pii_masking.py          # Main Python script
 ┣ 📜 README.md               # Project documentation
 ┗ 📜 requirements.txt        # Optional dependency list
```

---

## 🧰 Dependencies

* [PyTorch](https://pytorch.org/)
* [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index)

Install them using:

```bash
pip install torch transformers
```

---

## 🤖 Model Information

**Model:** [`iiiorg/piiranha-v1-detect-personal-information`](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information)

**Task:** Token Classification (Named Entity Recognition for PII detection)

**Supported Entity Labels:**

* `GIVENNAME`
* `SURNAME`
* `EMAIL_ADDRESS`
* `PHONE_NUMBER`
* `DATE_TIME`
* `LOCATION`
* and more...

---

## 🧭 Future Improvements

* Add support for **custom redaction tokens** (e.g., `[MASKED_NAME]`, `[MASKED_EMAIL]`)
* Build a **Streamlit or Flask web app** for user interaction
* Generate a **redacted JSON output** with entity metadata
* Integrate with **document redaction** (PDF, DOCX)

---

## 📄 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute it with attribution.

---

## 🧑‍💻 Author

**Pratham Raval**
Built with ❤️ using PyTorch and Hugging Face Transformers.

```

---

Would you like me to include **badges (Python, Hugging Face, License, GPU support)** and a **screenshot/demo output** section for a more polished GitHub presentation?
```
