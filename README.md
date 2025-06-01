# 🧠 SambaNova Medical Image Diagnosis Assistant

## Overview

The **SambaNova Medical Image Diagnosis Assistant** is a web-based AI tool that enables users to upload medical images and ask natural language questions related to those images. It leverages advanced large multimodal models (LMMs) via SambaNova’s API to generate medically-informed responses, providing a streamlined assistant for non-critical diagnostic insight or education.



---

## 🚀 Features

- 🔐 **Secure API Integration**: Uses `.env` to securely load the SambaNova API key.
- 🖼️ **Multi-Image Upload Support**: Upload and switch between multiple medical images.
- 🧠 **Image-Question AI Analysis**: Ask natural language questions about uploaded images.
- 📊 **Real-Time Results**: Displays AI response and response time.
- 💬 **Interactive Chat Interface**: Keeps a running history of user questions and assistant replies.
- ⚙️ **Prompt Engineering**: Uses structured prompts to encourage reliable, medically-cautious output.

---

## 🛠️ Tech Stack

| Component     | Technology             |
|---------------|-------------------------|
| Frontend      | Streamlit (Python UI)   |
| Backend       | Python, OpenAI SDK      |
| Image Handling| Pillow (PIL)            |
| Env Handling  | python-dotenv           |
| Deployment    | Streamlit CLI or Cloud  |

---

## 📂 Project Structure

```text
├── app.py                # Main application logic
├── requirements.txt      # Dependencies
├── .env                  # API Key storage (not committed to Git)
├── README.md             # This file
