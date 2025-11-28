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
        """Initialize the appropriate AI client based on provider."""
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
        """Get available models for the current provider."""
        try:
            if self.provider == "google":
                self.available_models = {
                    "gemini-2.5-pro	": {"id": "gemini-2.5-pro"},
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
            
            return dict(sorted(self.available_models.items()))  # Return models in alphabetical order
        
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

            # Generate response based on provider
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


            # Parse the response into Q&A pairs
            qa_pairs = []
            pattern = re.compile(
                r"(.*?)\?\s*\n(.*?)(?=\n\n|$)", 
                re.DOTALL
            )

            matches = pattern.findall(raw_qa)

            for q, a in matches:
                qa_pairs.append({
                    "question": q.strip(),
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

    def extract_text_from_pdf(self, pdf_file) -> List[str]:
        text_chunks = []
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                # Improved chunking to avoid breaking mid-sentence
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if len(para) > 1000:
                        chunks = [para[i:i+1000] for i in range(0, len(para), 1000)]
                        text_chunks.extend(chunks)
                    else:
                        text_chunks.append(para)
            return text_chunks
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return []

    def extract_text_from_docx(self, docx_file) -> List[str]:
        text_chunks = []
        try:
            doc = docx.Document(docx_file)
            for para in doc.paragraphs:
                if para.text.strip():
                    if len(para.text) > 1000:
                        chunks = [para.text[i:i+1000] for i in range(0, len(para.text), 1000)]
                        text_chunks.extend(chunks)
                    else:
                        text_chunks.append(para.text)
            return text_chunks
        except Exception as e:
            st.error(f"Error extracting text from DOCX: {str(e)}")
            return []

    def process_document(self, file, context: str, selected_model: str):
        if not file:
            st.error("Please upload a file")
            return

        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            file_extension = Path(file.name).suffix.lower()
            if file_extension == '.pdf':
                text_chunks = self.extract_text_from_pdf(file)
            elif file_extension in ['.doc', '.docx']:
                text_chunks = self.extract_text_from_docx(file)
            else:
                st.error("Unsupported file format")
                return

            if not text_chunks:
                st.error("No text could be extracted from the document")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(text_chunks):
                try:
                    qa_pairs = self.ai_manager.generate_qa_pairs(chunk, context, selected_model)
                    self.qa_pairs.extend(qa_pairs)

                    if i == 0 and qa_pairs:
                        st.write("### Sample Q&A Pairs")
                        for j, qa in enumerate(qa_pairs[:3], 1):
                            st.markdown(f"""
                            **Q{j}:** {qa['question']}
                            
                            **A{j}:** {qa['answer']}
                            
                            ---
                            """)

                    progress = (i + 1) / len(text_chunks)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing chunk {i + 1} of {len(text_chunks)}")

                except Exception as e:
                    st.error(f"Error processing chunk {i + 1}: {str(e)}")

            output_path = output_dir / self.output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "source_file": file.name,
                        "context": context,
                        "model": selected_model,
                        "total_qa_pairs": len(self.qa_pairs),
                        "total_chunks": len(text_chunks),
                        "average_pairs_per_chunk": len(self.qa_pairs)/len(text_chunks),
                        "generation_timestamp": str(datetime.datetime.now())
                    },
                    "qa_pairs": self.qa_pairs
                }, f, indent=2, ensure_ascii=False)

            status_text.text("Processing complete!")
            st.success(f"Q&A dataset saved to {output_path}")
            
            st.markdown("### Dataset Statistics")
            st.markdown(f"""
            - **Total Q&A pairs generated:** {len(self.qa_pairs)}
            - **Total text chunks processed:** {len(text_chunks)}
            - **Average Q&A pairs per chunk:** {len(self.qa_pairs)/len(text_chunks):.1f}
            """)

            if self.qa_pairs:
                st.markdown("### Random Sample Q&A Pairs")
                random_samples = random.sample(self.qa_pairs, min(3, len(self.qa_pairs)))
                for i, qa in enumerate(random_samples, 1):
                    st.markdown(f"""
                    **Q{i}:** {qa['question']}
                    
                    **A{i}:** {qa['answer']}
                    
                    ---
                    """)

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    st.set_page_config(page_title="Q&A Dataset Generator", layout="wide")
    st.title("Q&A Dataset Generator")

    load_dotenv()

    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}

    with st.sidebar:
        st.header("Settings")

        ai_provider = st.selectbox(
            "Select AI Provider",
            ["Ollama","Google"],
            key="ai_provider"
        )

        if st.button("Load model AI"):
            st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY")

            st.success("API keys loaded successfully!")

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
                context = st.text_area(
                    "Enter context for Q&A generation",
                    help="This context will be used to frame the questions appropriately",
                    height=100
                )

                output_file = st.text_input(
                    "Output JSON filename",
                    value="qa_dataset.json"
                )

            if st.button("Generate Q&A Dataset", type="primary"):
                if uploaded_file is not None and context:
                    processor = DatasetProcessor(ai_manager, output_file)
                    processor.process_document(uploaded_file, context, selected_model)
                else:
                    st.error("Please upload a file and provide context")
        else:
            st.error(f"No models available for {ai_provider}. Please check your API key and try again.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()