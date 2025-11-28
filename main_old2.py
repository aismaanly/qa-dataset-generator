import os
from dotenv import load_dotenv

import re
import streamlit as st
import PyPDF2
import json
import traceback
import google.generativeai as genai
import requests
import logging
from typing import Dict, List
from pathlib import Path
import docx
import datetime
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIManager:
    def __init__(self, provider: str):
        self.provider = provider.split()[0].lower()
        self.available_models = {}
        self.client = None
        self.setup_client()

    def load_prompt_template(self, path="src/prompt.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return ""

    def setup_client(self):
        try:
            if self.provider == "google":
                if not st.session_state.get('google_api_key'):
                    raise ValueError("Google API key not found")
                genai.configure(api_key=st.session_state.google_api_key)

            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434"
            
            logger.info(f"Successfully set up client for {self.provider}")

        except Exception as e:
            logger.error(f"Error setting up client for {self.provider}: {str(e)}")
            raise

    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        try:
            if self.provider == "google":
                self.available_models = {
                    "gemini-2.5-pro": {"id": "gemini-2.5-pro"},
                    "gemini-2.5-flash": {"id": "gemini-2.5-flash"}
                }

            elif self.provider == "ollama":
                try:
                    response = requests.get(f"{self.base_url}/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        self.available_models = {
                            model["name"]: {"id": model["name"]}
                            for model in models
                        }
                except Exception as e:
                    logger.error(f"Error getting Ollama models: {str(e)}")

            return dict(sorted(self.available_models.items()))
        
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return {}

    def generate_qa_pairs(self, text: str, context: str, model_id: str) -> List[Dict[str, str]]:
        try:
            template = self.load_prompt_template()

            prompt = f"""{template}

            Text:
            {text}

            Context:
            {context}
            """

            if self.provider == "google":
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt)
                raw_qa = response.text

            elif self.provider == "ollama":
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model_id, "prompt": prompt},
                    stream=True
                )

                raw_qa = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            raw_qa += data["response"]
                    except json.JSONDecodeError:
                        continue

            qa_pairs = []
            pattern = re.compile(r"(.*?)\?\s*\n(.*?)(?=\n\n|$)", re.DOTALL)
            matches = pattern.findall(raw_qa)

            for q, a in matches:
                qa_pairs.append({
                    "context": context,
                    "question": q.strip() + "?",
                    "answer": a.strip()
                })

            return qa_pairs

        except Exception as e:
            logger.error(f"Error generating QA pairs: {str(e)}")
            return []


class DatasetProcessor:
    def __init__(self, ai_manager, output_file='qa_dataset.json'):
        self.ai_manager = ai_manager
        self.output_file = output_file
        self.qa_pairs = []


    # Extract the whole document first
    def extract_full_text_pdf(self, pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            return full_text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""


    def extract_full_text_docx(self, docx_file) -> str:
        try:
            doc = docx.Document(docx_file)
            full_text = ""
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text += para.text + "\n"
            return full_text
        except Exception as e:
            st.error(f"Error extracting text from DOCX: {str(e)}")
            return ""


    # Split into BAB sections
    def split_by_bab(self, full_text: str) -> Dict[str, str]:
        pattern = r"(BAB\s+[IVXLC]+\s+[A-Z\s]+)"
        parts = re.split(pattern, full_text)

        result = {}
        current_bab = None

        for part in parts:
            part = part.strip()
            if part.startswith("BAB"):
                current_bab = part
                result[current_bab] = ""
            elif current_bab:
                result[current_bab] += part + "\n"

        return result


    def process_document(self, file, selected_model: str):
        if not file:
            st.error("Please upload a file")
            return

        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            file_extension = Path(file.name).suffix.lower()

            if file_extension == '.pdf':
                full_text = self.extract_full_text_pdf(file)

            elif file_extension in ['.doc', '.docx']:
                full_text = self.extract_full_text_docx(file)

            else:
                st.error("Unsupported file format")
                return

            if not full_text:
                st.error("No text could be extracted from the document")
                return

            bab_dict = self.split_by_bab(full_text)

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_bab = len(bab_dict)

            for i, (bab_title, bab_content) in enumerate(bab_dict.items()):
                qa_pairs = self.ai_manager.generate_qa_pairs(
                    text=bab_content,
                    context=bab_title,
                    model_id=selected_model
                )

                self.qa_pairs.extend(qa_pairs)

                progress = (i + 1) / total_bab
                progress_bar.progress(progress)
                status_text.text(f"Processing {bab_title} ({i+1}/{total_bab})")

            output_path = output_dir / self.output_file

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "source_file": file.name,
                        "model": selected_model,
                        "total_qa_pairs": len(self.qa_pairs),
                        "total_bab": total_bab,
                        "generation_timestamp": str(datetime.datetime.now())
                    },
                    "qa_pairs": self.qa_pairs
                }, f, indent=2, ensure_ascii=False)

            status_text.text("Processing complete!")
            st.success(f"Q&A dataset saved to {output_path}")

            st.markdown("### Dataset Statistics")
            st.markdown(f"- Total BAB: {total_bab}")
            st.markdown(f"- Total Q&A pairs: {len(self.qa_pairs)}")

            if self.qa_pairs:
                st.markdown("### Random Sample Q&A")
                sample = random.sample(self.qa_pairs, min(3, len(self.qa_pairs)))
                for s in sample:
                    st.markdown(f"**Context:** {s['context']}  \n**Q:** {s['question']}  \n**A:** {s['answer']}  \n---")

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            logger.error(traceback.format_exc())



def main():
    st.set_page_config(page_title="Q&A Dataset Generator", layout="wide")
    st.title("Q&A Dataset Generator")

    load_dotenv()

    with st.sidebar:
        st.header("Settings")

        ai_provider = st.selectbox(
            "Select AI Provider",
            ["Ollama", "Google"],
            key="ai_provider"
        )

        if st.button("Load model AI"):
            st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY")
            st.success("API key loaded!")

    try:
        ai_manager = AIManager(ai_provider)
        available_models = ai_manager.get_available_models()

        if available_models:
            col1, col2 = st.columns(2)

            with col1:
                selected_model = st.selectbox(
                    "Select AI Model",
                    options=list(available_models.keys()),
                    key="selected_model"
                )

                uploaded_file = st.file_uploader(
                    "Upload document",
                    type=["pdf", "doc", "docx"]
                )

            with col2:
                output_file = st.text_input(
                    "Output JSON filename",
                    value="qa_dataset.json"
                )

            if st.button("Generate Q&A Dataset", type="primary"):
                if uploaded_file is not None:
                    processor = DatasetProcessor(ai_manager, output_file)
                    processor.process_document(uploaded_file, selected_model)
                else:
                    st.error("Please upload a file")
        else:
            st.error(f"No models found for {ai_provider}. Check API key.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
