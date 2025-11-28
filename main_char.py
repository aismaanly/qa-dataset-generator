import os
import json
import re
import datetime
import logging
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import requests
import google.generativeai as genai

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# AI MANAGER
# =========================
class AIManager:
    def __init__(self, provider: str):
        self.provider = provider.lower()
        self.base_url = "http://localhost:11434"

        if self.provider == "google":
            genai.configure(api_key=st.session_state.get("google_api_key"))

    def get_available_models(self):
        if self.provider == "google":
            return {
                "gemini-2.5-flash": "gemini-2.5-flash",
                "gemini-2.5-pro": "gemini-2.5-pro"
            }

        if self.provider == "ollama":
            try:
                r = requests.get(f"{self.base_url}/api/tags", timeout=10)
                models = r.json().get("models", [])
                return {m["name"]: m["name"] for m in models}
            except Exception as e:
                logger.error(e)
                return {}

        return {}

    def load_prompt(self, path="src/prompt.txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def generate(self, text: str, context: str, model_id: str):
        prompt_template = self.load_prompt()

        prompt = f"""{prompt_template}

Text:
{text}

Context:
{context}
"""

        try:
            if self.provider == "google":
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt)
                return response.text.strip() if response and response.text else ""

            if self.provider == "ollama":
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_id,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9
                        }
                    },
                    timeout=300
                )
                return response.json().get("response", "").strip()

        except Exception as e:
            logger.error(f"❌ AI Error: {e}")

        return ""


# =========================
# DATASET PROCESSOR
# =========================
class DatasetProcessor:
    def __init__(self, ai_manager, output_file):
        self.ai_manager = ai_manager
        self.output_file = output_file
        self.qa_pairs = []

    def parse_qa(self, raw_text):
        pattern = re.compile(
            r"Context:\s*(.*?)\nQuestion:\s*(.*?)\nAnswer:\s*(.*?)(?=\nContext:|\Z)",
            re.DOTALL
        )

        results = []
        for ctx, q, a in pattern.findall(raw_text):
            results.append({
                "context": ctx.strip(),
                "question": q.strip(),
                "answer": a.strip()
            })
        return results

    def process_json(self, json_path: Path, model_id: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            st.error("❌ Format JSON harus berupa LIST objek BAB")
            return

        progress = st.progress(0)
        status = st.empty()
        total = len(data)

        MAX_CHARS = 8000

        for i, bab in enumerate(data):
            if not isinstance(bab, dict):
                continue

            judul_bab = str(bab.get("judul_bab", "")).strip()
            if not judul_bab:
                logger.warning("⚠️ BAB tanpa judul dilewati")
                continue

            texts = []
            for pasal in bab.get("pasal", []):
                if isinstance(pasal, dict):
                    for item in pasal.get("detail", []):
                        if isinstance(item, str) and item.strip():
                            texts.append(item.strip())

            if not texts:
                logger.warning(f"⚠️ BAB kosong: {judul_bab}")
                continue

            combined_text = "\n".join(texts)

            if len(combined_text) > MAX_CHARS:
                logger.warning(f"⚠️ BAB {judul_bab} terlalu panjang, dipadatkan")
                combined_text = combined_text[:MAX_CHARS]

            raw_output = self.ai_manager.generate(
                text=combined_text,
                context=judul_bab,
                model_id=model_id
            )

            if not raw_output:
                logger.error(f"❌ Output kosong untuk BAB: {judul_bab}")
                continue

            qa = self.parse_qa(raw_output)

            if not qa:
                logger.warning(f"⚠️ Tidak ada QA dihasilkan untuk BAB: {judul_bab}")
                continue

            logger.info(f"✅ {len(qa)} QA dihasilkan untuk BAB: {judul_bab}")
            self.qa_pairs.extend(qa)

            progress.progress((i + 1) / total)
            status.text(f"Processing {judul_bab} ({i+1}/{total})")

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / self.output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "source_file": json_path.name,
                    "model": model_id,
                    "total_qa_pairs": len(self.qa_pairs),
                    "generation_timestamp": str(datetime.datetime.now())
                },
                "qa_pairs": self.qa_pairs
            }, f, indent=2, ensure_ascii=False)

        st.success(f"✅ Dataset berhasil dibuat: {output_path}")


# =========================
# STREAMLIT APP
# =========================
def main():
    st.set_page_config(page_title="Synthetic Q&A Generator", layout="wide")
    st.title("Synthetic Q&A Dataset Generator (PERTOR JSON)")

    load_dotenv()

    with st.sidebar:
        provider = st.selectbox("AI Provider", ["ollama", "google"])
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY")

    data_dir = Path("data")
    json_files = sorted(data_dir.glob("*.json"))

    if not json_files:
        st.error("❌ Tidak ada file JSON di folder data/")
        return

    selected_json = st.selectbox(
        "Pilih file JSON",
        options=json_files,
        format_func=lambda x: x.name
    )

    ai_manager = AIManager(provider)
    models = ai_manager.get_available_models()

    if not models:
        st.error("❌ Model tidak ditemukan")
        return

    model_id = st.selectbox("Pilih Model AI", list(models.keys()))

    output_file = st.text_input(
        "Nama file output",
        value=f"qa_{selected_json.stem}.json"
    )

    if st.button("Generate Synthetic Q&A", type="primary"):
        processor = DatasetProcessor(ai_manager, output_file)
        processor.process_json(selected_json, model_id)


if __name__ == "__main__":
    main()
