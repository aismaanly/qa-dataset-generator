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

from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger(__name__)


# =========================
# AI MANAGER
# =========================
class AIManager:
    def __init__(self, provider: str):
        self.provider = provider.lower()
        self.base_url = "http://localhost:11434"

        if self.provider == "google":
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
                logger.error(f"Ollama error: {e}")
                return {}

        return {}

    def load_prompt(self, path="src/prompt.txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def generate(self, text: str, context: str, model_id: str) -> str:
        prompt = f"""{self.load_prompt()}

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
                            "temperature": 0.4,
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

        # Teknik chunk
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ";"]
        )

    # ✅ FAIL-SOFT PARSER
    def parse_qa(self, raw_text: str):
        results = []

        blocks = re.split(r"\n(?=Context:)", raw_text.strip())
        for block in blocks:
            ctx = re.search(r"Context:\s*(.+)", block)
            q = re.search(r"Question:\s*(.+)", block)
            a = re.search(r"Answer:\s*(.+)", block, re.DOTALL)

            if ctx and q and a:
                results.append({
                    "context": ctx.group(1).strip(),
                    "question": q.group(1).strip(),
                    "answer": a.group(1).strip()
                })

        return results

    def process_json(self, json_path: Path, model_id: str):
        data = json.loads(json_path.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            st.error("❌ JSON harus berbentuk LIST BAB")
            return

        progress = st.progress(0.0)
        status = st.empty()

        for i, bab in enumerate(data):
            judul_bab = str(bab.get("judul_bab", "")).strip()
            if not judul_bab:
                continue

            texts = []
            for pasal in bab.get("pasal", []):
                if isinstance(pasal, dict):
                    texts.extend(
                        [t.strip() for t in pasal.get("detail", []) if isinstance(t, str)]
                    )

            if not texts:
                logger.warning(f"⚠️ BAB kosong: {judul_bab}")
                continue

            docs = self.text_splitter.create_documents(["\n".join(texts)])
            bab_qa = 0

            for idx, doc in enumerate(docs):
                output = self.ai_manager.generate(
                    doc.page_content,
                    judul_bab,
                    model_id
                )

                if not output:
                    continue

                qa = self.parse_qa(output)
                if qa:
                    self.qa_pairs.extend(qa)
                    bab_qa += len(qa)
                else:
                    logger.info(f"ℹ️ Chunk {idx+1} BAB {judul_bab} tanpa QA")

            if bab_qa == 0:
                logger.warning(f"⚠️ Tidak ada QA dihasilkan untuk BAB: {judul_bab}")
            else:
                logger.info(f"✅ {bab_qa} QA dihasilkan untuk BAB: {judul_bab}")

            progress.progress((i + 1) / len(data))
            status.text(f"Memproses {judul_bab} ({i+1}/{len(data)})")

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / self.output_file
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "source": json_path.name,
                    "model": model_id,
                    "total_qa_pairs": len(self.qa_pairs),
                    "generated_at": datetime.datetime.now().isoformat()
                },
                "qa_pairs": self.qa_pairs
            }, f, indent=2, ensure_ascii=False)

        st.success(f"✅ Dataset berhasil dibuat: {output_path}")


# =========================
# STREAMLIT APP
# =========================
def main():
    st.set_page_config("Synthetic Q&A Generator", layout="wide")
    st.title("📘 Synthetic Q&A Dataset Generator (PERTOR JSON)")

    load_dotenv()

    provider = st.sidebar.selectbox("AI Provider", ["ollama", "google"])

    data_dir = Path("data")
    files = list(data_dir.glob("*.json"))
    if not files:
        st.error("❌ Folder data/ tidak berisi JSON")
        return

    json_file = st.selectbox("Pilih File JSON", files, format_func=lambda x: x.name)

    ai = AIManager(provider)
    models = ai.get_available_models()

    if not models:
        st.error("❌ Model tidak tersedia")
        return

    model_id = st.selectbox("Pilih Model", list(models.keys()))

    output_file = st.text_input(
        "Nama file output",
        f"qa_{json_file.stem}.json"
    )

    if st.button("🚀 Generate Dataset", type="primary"):
        processor = DatasetProcessor(ai, output_file)
        processor.process_json(json_file, model_id)


if __name__ == "__main__":
    main()
