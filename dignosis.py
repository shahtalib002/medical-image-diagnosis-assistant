# # # import os
# # # import io
# # # import base64
# # # import logging
# # # import requests
# # # from PIL import Image
# # # import streamlit as st
# # # import torch
# # # from dotenv import load_dotenv
# # # from transformers import CLIPProcessor, CLIPModel
# # # import openai  # OpenAI-compatible SDK for SambaNova

# # # # Set up logging
# # # logging.basicConfig(
# # #     filename="app_debug.log",
# # #     filemode="a",
# # #     format="%(asctime)s - %(levelname)s - %(message)s",
# # #     level=logging.INFO
# # # )

# # # # Load environment variables
# # # load_dotenv()
# # # SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# # # # Ensure key is set
# # # if not SAMBANOVA_API_KEY:
# # #     st.error("SambaNova API key not found. Please check your .env file.")
# # #     st.stop()

# # # # Load CLIP model to verify medical image
# # # @st.cache_resource
# # # def load_medical_filter():
# # #     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # #     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # #     return processor, model

# # # processor, model = load_medical_filter()

# # # # Labels for detecting medical images
# # # MEDICAL_LABELS = [
# # #     "X-ray", "CT scan", "MRI", "ultrasound", "medical scan",
# # #     "radiology image", "chest x-ray", "brain scan", "medical imaging"
# # # ]

# # # def is_medical_image(image: Image.Image) -> bool:
# # #     inputs = processor(text=MEDICAL_LABELS, images=image, return_tensors="pt", padding=True)
# # #     outputs = model(**inputs)
# # #     logits_per_image = outputs.logits_per_image
# # #     probs = logits_per_image.softmax(dim=1)
# # #     pred_idx = torch.argmax(probs, dim=1).item()
# # #     pred_label = MEDICAL_LABELS[pred_idx]
# # #     confidence = probs[0][pred_idx].item()
# # #     logging.info(f"Predicted label: {pred_label} (confidence: {confidence:.2f})")
# # #     return pred_label in MEDICAL_LABELS and confidence > 0.25

# # # # Query SambaNova API using OpenAI-compatible client
# # # def query_sambanova_api(image: Image.Image) -> str:
# # #     buffered = io.BytesIO()
# # #     image.save(buffered, format="JPEG")
# # #     image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

# # #     # SambaNova-compatible OpenAI client
# # #     client = openai.OpenAI(
# # #         base_url="https://api.sambanova.ai/v1",
# # #         api_key=SAMBANOVA_API_KEY
# # #     )

# # #     try:
# # #         response = client.chat.completions.create(
# # #             model="Llama-4-Maverick-17B-128E-Instruct",
# # #             messages=[
# # #                 {
# # #                     "role": "user",
# # #                     "content": [
# # #                         {
# # #                             "type": "text",
# # #                             "text": "You are a professional radiologist. Analyze the uploaded medical image and provide a concise yet medically accurate diagnosis."
# # #                         },
# # #                         {
# # #                             "type": "image_url",
# # #                             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
# # #                         }
# # #                     ]
# # #                 }
# # #             ]
# # #         )
# # #         return response.choices[0].message.content
# # #     except Exception as e:
# # #         logging.error(f"API call failed: {e}")
# # #         return f"API Error: {e}"

# # # # Streamlit UI
# # # st.set_page_config(page_title="Medical Diagnosis Assistant", layout="wide")
# # # st.title("🧠 Medical Image Diagnosis Assistant")
# # # st.write("Upload medical images such as X-rays, CT scans, or MRIs to receive AI-assisted diagnosis.")

# # # uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# # # if uploaded_files:
# # #     for idx, file in enumerate(uploaded_files):
# # #         image = Image.open(file)
# # #         col1, col2 = st.columns([1, 2])

# # #         with col1:
# # #             st.image(image, caption=f"Image {idx+1}: {file.name}", use_container_width=True)

# # #         with col2:
# # #             if is_medical_image(image):
# # #                 st.info("✅ Recognized as a medical image. Proceeding with diagnosis...")
# # #                 with st.spinner("Diagnosing via SambaNova API..."):
# # #                     diagnosis = query_sambanova_api(image)
# # #                 st.success("Diagnosis Complete")
# # #                 st.markdown(f"**Diagnosis:** {diagnosis}")
# # #             else:
# # #                 st.warning(f"🚫 The image '{file.name}' is not recognized as medical (e.g., X-ray, MRI). Skipping.")
# # #                 logging.info(f"Rejected image: {file.name} - classified as non-medical")
 
# # import os
# # import io
# # import base64
# # import logging
# # import requests
# # from PIL import Image
# # import streamlit as st
# # import torch
# # from dotenv import load_dotenv
# # from transformers import CLIPProcessor, CLIPModel
# # import openai

# # # Load .env and API key
# # load_dotenv()
# # SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# # # Setup logging
# # logging.basicConfig(filename="app_debug.log", level=logging.INFO)

# # # Setup OpenAI-compatible SambaNova client
# # client = openai.OpenAI(
# #     base_url="https://api.sambanova.ai/v1",
# #     api_key=SAMBANOVA_API_KEY
# # )

# # # Load CLIP for image classification
# # @st.cache_resource
# # def load_medical_filter():
# #     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# #     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# #     return processor, model

# # processor, model = load_medical_filter()

# # MEDICAL_LABELS = [
# #     "X-ray", "CT scan", "MRI", "ultrasound", "medical scan",
# #     "radiology image", "chest x-ray", "brain scan", "medical imaging"
# # ]

# # def is_medical_image(image: Image.Image) -> bool:
# #     inputs = processor(text=MEDICAL_LABELS, images=image, return_tensors="pt", padding=True)
# #     outputs = model(**inputs)
# #     probs = outputs.logits_per_image.softmax(dim=1)
# #     pred_idx = torch.argmax(probs, dim=1).item()
# #     pred_label = MEDICAL_LABELS[pred_idx]
# #     confidence = probs[0][pred_idx].item()
# #     return pred_label in MEDICAL_LABELS and confidence > 0.25

# # def image_to_base64(image: Image.Image) -> str:
# #     buf = io.BytesIO()
# #     image.save(buf, format="JPEG")
# #     return base64.b64encode(buf.getvalue()).decode("utf-8")

# # # Diagnose image
# # def query_diagnosis(image_base64: str, style="professional") -> str:
# #     prompt = {
# #         "professional": "You are a professional radiologist. Analyze the uploaded medical image and provide a concise yet medically accurate diagnosis.",
# #         "detailed": "You are a medical expert. Please analyze this image in detail, explaining key observations in plain language.",
# #         "layman": "Explain this medical image in a way a non-medical person can understand."
# #     }.get(style, "professional")

# #     response = client.chat.completions.create(
# #         model="Llama-4-Maverick-17B-128E-Instruct",
# #         messages=[{
# #             "role": "user",
# #             "content": [
# #                 {"type": "text", "text": prompt},
# #                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
# #             ]
# #         }]
# #     )
# #     return response.choices[0].message.content

# # # Handle chat queries about diagnosis
# # def query_chat_followup(history):
# #     full_chat = [
# #         {"role": "system", "content": "You are a helpful and medically accurate AI assistant. Respond only in plain natural language without using code or tools."}
# #     ] + history

# #     response = client.chat.completions.create(
# #         model="Llama-4-Maverick-17B-128E-Instruct",
# #         messages=full_chat
# #     )
# #     return response.choices[0].message.content


# # # Streamlit UI
# # st.set_page_config("Medical AI Assistant", layout="wide")
# # st.title("🧠 Medical Image Diagnosis Assistant")

# # # Sidebar
# # st.sidebar.title("⚙️ Options")
# # style = st.sidebar.selectbox("Diagnosis Style", ["professional", "detailed", "layman"])
# # auto_questions = st.sidebar.checkbox("Auto-suggest questions", value=True)
# # if st.sidebar.button("Clear Chat"):
# #     if "chat_history" in st.session_state:
# #         st.session_state.pop("chat_history")
# #     st.rerun()


# # st.sidebar.markdown("---")
# # st.sidebar.info("Upload medical images like X-rays or MRIs to receive an AI-powered diagnosis. You can then ask follow-up questions in natural language.")

# # # Session state for chat history
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []

# # uploaded_files = st.file_uploader("📤 Upload medical images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# # if uploaded_files:
# #     for idx, file in enumerate(uploaded_files):
# #         image = Image.open(file)
# #         col1, col2 = st.columns([1, 2])

# #         with col1:
# #             st.image(image, caption=f"Image {idx+1}: {file.name}", use_container_width=True)

# #         with col2:
# #             if is_medical_image(image):
# #                 st.success("✅ Medical image detected. Diagnosing...")
# #                 base64_img = image_to_base64(image)
# #                 diagnosis = query_diagnosis(base64_img, style=style)
# #                 st.markdown(f"**Diagnosis:** {diagnosis}")
# #                 st.session_state.chat_history = [
# #                     {"role": "user", "content": "Explain the medical image."},
# #                     {"role": "assistant", "content": diagnosis}
# #                 ]
# #                 if auto_questions:
# #                     st.markdown("💬 Suggested questions:")
# #                     st.button("What does this condition mean?", on_click=lambda: st.session_state.update({"last_q": "What does this condition mean?"}))
# #                     st.button("Is this a serious issue?", on_click=lambda: st.session_state.update({"last_q": "Is this a serious issue?"}))
# #             else:
# #                 st.warning("⚠️ Not recognized as a medical image.")

# # # Chat input
# # st.markdown("### 💬 Ask a question about the diagnosis")
# # user_question = st.text_input("Type your question here:")

# # if user_question or st.session_state.get("last_q"):
# #     question = user_question if user_question else st.session_state.pop("last_q")
# #     st.session_state.chat_history.append({"role": "user", "content": question})
# #     with st.spinner("Generating response..."):
# #         response = query_chat_followup(st.session_state.chat_history)
# #     st.session_state.chat_history.append({"role": "assistant", "content": response})
# #     st.markdown(f"**AI:** {response}")


# import os
# import io
# import base64
# import logging
# import streamlit as st
# from PIL import Image
# import torch
# from dotenv import load_dotenv
# from transformers import CLIPProcessor, CLIPModel
# import openai

# # Load environment variables
# load_dotenv()
# SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# # Logging
# logging.basicConfig(filename="app_debug.log", level=logging.INFO)

# # OpenAI-compatible SambaNova client
# client = openai.OpenAI(
#     base_url="https://api.sambanova.ai/v1",
#     api_key=SAMBANOVA_API_KEY
# )

# # Load CLIP model to identify medical images
# @st.cache_resource
# def load_medical_filter():
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     return processor, model

# processor, model = load_medical_filter()

# MEDICAL_LABELS = [
#     "X-ray", "CT scan", "MRI", "ultrasound", "medical scan",
#     "radiology image", "chest x-ray", "brain scan", "medical imaging"
# ]

# # Medical image checker
# def is_medical_image(image: Image.Image) -> bool:
#     inputs = processor(text=MEDICAL_LABELS, images=image, return_tensors="pt", padding=True)
#     outputs = model(**inputs)
#     probs = outputs.logits_per_image.softmax(dim=1)
#     pred_idx = torch.argmax(probs, dim=1).item()
#     pred_label = MEDICAL_LABELS[pred_idx]
#     confidence = probs[0][pred_idx].item()
#     logging.info(f"Prediction: {pred_label} (Confidence: {confidence:.2f})")
#     return pred_label in MEDICAL_LABELS and confidence > 0.35

# # Convert image to base64 safely
# def image_to_base64(image: Image.Image) -> str:
#     if image.mode in ("RGBA", "LA", "P"):
#         image = image.convert("RGB")
#     buf = io.BytesIO()
#     image.save(buf, format="JPEG")
#     return base64.b64encode(buf.getvalue()).decode("utf-8")

# # Query diagnosis
# def query_diagnosis(image_base64: str, style="professional") -> str:
#     prompt = {
#         "professional": "You are a professional radiologist. Analyze the uploaded medical image and provide a concise yet medically accurate diagnosis.",
#         "detailed": "You are a medical expert. Please analyze this image in detail, explaining key observations in plain language.",
#         "layman": "Explain this medical image in a way a non-medical person can understand."
#     }.get(style, "professional")

#     response = client.chat.completions.create(
#         model="Llama-4-Maverick-17B-128E-Instruct",
#         messages=[{
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
#             ]
#         }]
#     )
#     return response.choices[0].message.content

# # Query follow-up chat
# def query_chat_followup(history):
#     full_chat = [
#         {"role": "system", "content": "You are a helpful and medically accurate AI assistant. Respond only in plain natural language without using code or tools."}
#     ] + history

#     response = client.chat.completions.create(
#         model="Llama-4-Maverick-17B-128E-Instruct",
#         messages=full_chat
#     )
#     return response.choices[0].message.content

# # Streamlit UI
# st.set_page_config(page_title="Medical Diagnosis Assistant", layout="wide")
# st.title("🧠 Medical Image Diagnosis Assistant")
# st.write("Upload medical images (X-ray, MRI, CT, etc.) to receive AI-assisted diagnosis. You can also ask follow-up questions.")

# # Sidebar options
# st.sidebar.title("⚙️ Options")
# style = st.sidebar.selectbox("Diagnosis Style", ["professional", "detailed", "layman"])
# auto_questions = st.sidebar.checkbox("Auto-suggest questions", value=True)
# if st.sidebar.button("Clear Chat"):
#     if "chat_history" in st.session_state:
#         st.session_state.pop("chat_history")
#     st.rerun()
# st.sidebar.markdown("---")
# st.sidebar.info("Only upload real medical images. Screenshots or non-medical content will be rejected.")

# # Session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # File uploader
# uploaded_files = st.file_uploader("📤 Upload medical images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# if uploaded_files:
#     for idx, file in enumerate(uploaded_files):
#         image = Image.open(file)
#         col1, col2 = st.columns([1, 2])

#         with col1:
#             st.image(image, caption=f"Image {idx+1}: {file.name}", use_container_width=True)

#         with col2:
#             if is_medical_image(image):
#                 st.success("✅ Medical image detected. Diagnosing...")
#                 base64_img = image_to_base64(image)
#                 diagnosis = query_diagnosis(base64_img, style=style)
#                 st.markdown(f"**Diagnosis:** {diagnosis}")
#                 st.session_state.chat_history = [
#                     {"role": "user", "content": "Explain the medical image."},
#                     {"role": "assistant", "content": diagnosis}
#                 ]
#                 if auto_questions:
#                     st.markdown("💬 Suggested Questions:")
#                     if st.button("What does this condition mean?"):
#                         st.session_state["last_q"] = "What does this condition mean?"
#                     if st.button("Is this a serious issue?"):
#                         st.session_state["last_q"] = "Is this a serious issue?"
#             else:
#                 st.warning("⚠️ This image is not recognized as a valid medical image. Skipping diagnosis.")

# # Chat input
# st.markdown("### 💬 Ask a question about the diagnosis")
# user_question = st.text_input("Type your question here:")

# if user_question or st.session_state.get("last_q"):
#     question = user_question if user_question else st.session_state.pop("last_q")
#     st.session_state.chat_history.append({"role": "user", "content": question})
#     with st.spinner("Answering your question..."):
#         response = query_chat_followup(st.session_state.chat_history)
#     st.session_state.chat_history.append({"role": "assistant", "content": response})
#     st.markdown(f"**AI:** {response}")
 
# import os
# import io
# import base64
# import logging
# import streamlit as st
# from PIL import Image
# import torch
# from dotenv import load_dotenv
# from transformers import CLIPProcessor, CLIPModel
# import openai
# from fpdf import FPDF
# import io

# # Load environment variables
# load_dotenv()
# SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# # Logging
# logging.basicConfig(filename="app_debug.log", level=logging.INFO)

# # OpenAI-compatible SambaNova client
# client = openai.OpenAI(
#     base_url="https://api.sambanova.ai/v1",
#     api_key=SAMBANOVA_API_KEY
# )

# # Load CLIP model to identify medical images
# @st.cache_resource
# def load_medical_filter():
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     return processor, model

# processor, model = load_medical_filter()

# MEDICAL_LABELS = [
#     "X-ray", "CT scan", "MRI", "ultrasound", "medical scan",
#     "radiology image", "chest x-ray", "brain scan", "medical imaging"
# ]

# # Image checker

# def is_medical_image(image: Image.Image) -> bool:
#     inputs = processor(text=MEDICAL_LABELS, images=image, return_tensors="pt", padding=True)
#     outputs = model(**inputs)
#     probs = outputs.logits_per_image.softmax(dim=1)
#     pred_idx = torch.argmax(probs, dim=1).item()
#     pred_label = MEDICAL_LABELS[pred_idx]
#     confidence = probs[0][pred_idx].item()
#     logging.info(f"Prediction: {pred_label} (Confidence: {confidence:.2f})")
#     return pred_label in MEDICAL_LABELS and confidence > 0.35

# # Convert image to base64

# def image_to_base64(image: Image.Image) -> str:
#     if image.mode in ("RGBA", "LA", "P"):
#         image = image.convert("RGB")
#     buf = io.BytesIO()
#     image.save(buf, format="JPEG")
#     return base64.b64encode(buf.getvalue()).decode("utf-8")

# # Diagnosis query

# def query_diagnosis(image_base64: str, style="professional") -> str:
#     prompt = {
#         "professional": "You are a professional radiologist. Analyze the uploaded medical image and provide a diagnosis in plain English. Do not use any tools, functions, or code in your answer. Just respond in natural language.",
#         "detailed": "You are a medical expert. Please analyze this image in detail using plain English. Do not use any tools or code.",
#         "layman": "Explain this medical image in simple language. Avoid using tools, code, or medical jargon."
#     }.get(style, "professional")

#     response = client.chat.completions.create(
#         model="Llama-4-Maverick-17B-128E-Instruct",
#         messages=[{
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
#             ]
#         }]
#     )
#     return response.choices[0].message.content


# # Chat follow-up

# def query_chat_followup(history):
#     full_chat = [
#         {"role": "system", "content": "You are a helpful and medically accurate AI assistant. Respond only in plain natural language without using code or tools."}
#     ] + history

#     response = client.chat.completions.create(
#         model="Llama-4-Maverick-17B-128E-Instruct",
#         messages=full_chat
#     )
#     return response.choices[0].message.content
# from fpdf import FPDF

# def generate_pdf(image: Image.Image, diagnosis: str, qa: list) -> bytes:
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", "B", 14)
#     pdf.cell(200, 10, "Medical Diagnosis Report", ln=True, align="C")

#     # Save and insert image
#     temp_path = "temp_image.jpg"
#     image.save(temp_path)
#     pdf.image(name=temp_path, x=60, y=30, w=90)
#     pdf.ln(95)

#     # Add diagnosis
#     pdf.set_font("Arial", "", 12)
#     pdf.multi_cell(0, 10, f"Diagnosis: {diagnosis}")

#     # Add Chat History if available
#     if qa:
#         pdf.set_font("Arial", "B", 12)
#         pdf.cell(0, 10, "Chat History:", ln=True)
#         pdf.set_font("Arial", "", 12)
#         for q, a in qa:
#             pdf.multi_cell(0, 8, f"Q: {q}\nA: {a}")

#     # Return as bytes
#     output = io.BytesIO()
#     pdf.output(name=output, dest='S')
#     output.seek(0)
#     return output.read()





# # Streamlit UI setup
# st.set_page_config(page_title="Medical Diagnosis Assistant", layout="wide")
# st.title("🧠 Medical Image Diagnosis Assistant")
# st.write("Upload medical images (X-ray, MRI, CT) and get AI-assisted diagnosis. Then ask follow-up questions per image.")

# # Sidebar options
# st.sidebar.title("⚙️ Options")
# style = st.sidebar.selectbox("Diagnosis Style", ["professional", "detailed", "layman"])
# if st.sidebar.button("Clear All"):
#     st.session_state.clear()
#     st.rerun()

# # Session state setup
# if "images" not in st.session_state:
#     st.session_state.images = {}

# # Upload
# uploaded_files = st.file_uploader("📤 Upload medical images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# if uploaded_files:
#     for file in uploaded_files:
#         image = Image.open(file)
#         if not is_medical_image(image):
#             st.warning(f"{file.name} is not recognized as a valid medical image. Skipping.")
#             continue

#         base64_img = image_to_base64(image)
#         diagnosis = query_diagnosis(base64_img, style=style)

#         st.session_state.images[file.name] = {
#             "image": image,
#             "base64": base64_img,
#             "diagnosis": diagnosis,
#             "chat": [
#                 {"role": "user", "content": "Explain the medical image."},
#                 {"role": "assistant", "content": diagnosis}
#             ]
#         }

# # Chat per image
# if st.session_state.images:
#     selected_image = st.selectbox("📂 Select an image to view and chat about:", list(st.session_state.images.keys()))
#     data = st.session_state.images[selected_image]

#     st.image(data["image"], caption=selected_image, use_container_width=True)
#     st.markdown(f"**Diagnosis:** {data['diagnosis']}")
# # Prepare chat history for PDF
# chat_history = []
# for entry in data["chat"]:
#     if entry["role"] == "user":
#         user_question = entry["content"]
#         # Try to find the next assistant response
#         next_idx = data["chat"].index(entry) + 1
#         if next_idx < len(data["chat"]) and data["chat"][next_idx]["role"] == "assistant":
#             assistant_reply = data["chat"][next_idx]["content"]
#             chat_history.append((user_question, assistant_reply))

# # Add Download Report button
# if st.button("📄 Generate PDF Report"):
#     pdf_bytes = generate_pdf(data["image"], data["diagnosis"], chat_history)
#     st.download_button("Download Diagnosis Report", data=pdf_bytes, file_name=f"diagnosis_{selected_image}.pdf", mime="application/pdf")

#     user_q = st.text_input("💬 Ask a follow-up question:")
#     if user_q:
#         data["chat"].append({"role": "user", "content": user_q})
#         with st.spinner("Answering..."):
#             reply = query_chat_followup(data["chat"])
#         data["chat"].append({"role": "assistant", "content": reply})
#         st.markdown(f"**AI:** {reply}")

#     # Display chat history
#     if len(data["chat"]) > 2:
#         with st.expander("🗂️ View full chat history"):
#             for entry in data["chat"][2:]:
#                 st.markdown(f"**{entry['role'].capitalize()}**: {entry['content']}")
import os
import io
import base64
import logging
import streamlit as st
from PIL import Image
import torch
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import openai
from fpdf import FPDF

# Load environment variables
load_dotenv()
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# Logging
logging.basicConfig(filename="app_debug.log", level=logging.INFO)

# OpenAI-compatible SambaNova client
client = openai.OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=SAMBANOVA_API_KEY
)

# Load CLIP model to identify medical images
@st.cache_resource
def load_medical_filter():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

processor, model = load_medical_filter()

MEDICAL_LABELS = [
    "X-ray", "CT scan", "MRI", "ultrasound", "medical scan",
    "radiology image", "chest x-ray", "brain scan", "medical imaging"
]

def is_medical_image(image: Image.Image) -> bool:
    inputs = processor(text=MEDICAL_LABELS, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_label = MEDICAL_LABELS[pred_idx]
    confidence = probs[0][pred_idx].item()
    logging.info(f"Prediction: {pred_label} (Confidence: {confidence:.2f})")
    return pred_label in MEDICAL_LABELS and confidence > 0.35

def image_to_base64(image: Image.Image) -> str:
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def query_diagnosis(image_base64: str, style="professional") -> str:
    prompt = {
        "professional": "You are a professional radiologist. Analyze the uploaded medical image and provide a diagnosis in plain English.",
        "detailed": "You are a medical expert. Please analyze this image in detail using plain English.",
        "layman": "Explain this medical image in simple language."
    }.get(style, "professional")

    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }]
    )
    return response.choices[0].message.content

def query_chat_followup(history):
    full_chat = [
        {"role": "system", "content": "You are a helpful and medically accurate AI assistant."}
    ] + history

    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct",
        messages=full_chat
    )
    return response.choices[0].message.content

def generate_pdf(image: Image.Image, diagnosis: str, qa: list) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Medical Diagnosis Report", ln=True, align="C")

    # Save and insert the image
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    pdf.image(name=temp_path, x=60, y=30, w=90)
    pdf.ln(95)

    # Add diagnosis
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Diagnosis: {diagnosis}")

    # Add chat history
    if qa:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Chat History:", ln=True)
        pdf.set_font("Arial", "", 12)
        for q, a in qa:
            pdf.multi_cell(0, 8, f"Q: {q}\nA: {a}")

    # Write PDF to bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # Output as string, then encode to bytes
    return pdf_bytes

# Streamlit UI
st.set_page_config(page_title="Medical Diagnosis Assistant", layout="wide")
st.title("🧠 Medical Image Diagnosis Assistant")
st.write("Upload medical images (X-ray, MRI, CT) and get AI-assisted diagnosis. Then ask follow-up questions.")

st.sidebar.title("⚙️ Options")
style = st.sidebar.selectbox("Diagnosis Style", ["professional", "detailed", "layman"])
if st.sidebar.button("Clear All"):
    st.session_state.clear()
    st.rerun()

if "images" not in st.session_state:
    st.session_state.images = {}

uploaded_files = st.file_uploader("📤 Upload medical images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        if not is_medical_image(image):
            st.warning(f"{file.name} is not recognized as a valid medical image. Skipping.")
            continue

        base64_img = image_to_base64(image)
        diagnosis = query_diagnosis(base64_img, style=style)

        st.session_state.images[file.name] = {
            "image": image,
            "base64": base64_img,
            "diagnosis": diagnosis,
            "chat": [
                {"role": "user", "content": "Explain the medical image."},
                {"role": "assistant", "content": diagnosis}
            ]
        }

if st.session_state.images:
    selected_image = st.selectbox("📂 Select an image to view and chat about:", list(st.session_state.images.keys()))
    data = st.session_state.images[selected_image]

    st.image(data["image"], caption=selected_image, use_container_width=True)
    st.markdown(f"**Diagnosis:** {data['diagnosis']}")

    # Chat input
    user_q = st.text_input("💬 Ask a follow-up question:")
    if user_q:
        data["chat"].append({"role": "user", "content": user_q})
        with st.spinner("Answering..."):
            reply = query_chat_followup(data["chat"])
        data["chat"].append({"role": "assistant", "content": reply})
        st.markdown(f"**AI:** {reply}")

    # Download report
    chat_history = []
    for i in range(0, len(data["chat"]) - 1):
        if data["chat"][i]["role"] == "user" and data["chat"][i + 1]["role"] == "assistant":
            chat_history.append((data["chat"][i]["content"], data["chat"][i + 1]["content"]))

    if st.button("📄 Generate PDF Report"):
        pdf_bytes = generate_pdf(data["image"], data["diagnosis"], chat_history)
        st.download_button("Download Diagnosis Report", data=pdf_bytes, file_name=f"diagnosis_{selected_image}.pdf", mime="application/pdf")

    # Show chat history
    if len(data["chat"]) > 2:
        with st.expander("🗂️ View full chat history"):
            for entry in data["chat"][2:]:
                st.markdown(f"**{entry['role'].capitalize()}**: {entry['content']}")

