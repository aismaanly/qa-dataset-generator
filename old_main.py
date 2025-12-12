import json
import re
import datetime
import logging
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import requests
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
# AI MANAGER (Ollama only)
# =========================
class AIManager:
    def __init__(self, provider: str):
        self.provider = provider.lower()
        self.base_url = "http://localhost:11434"

    def get_available_models(self):
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
            # ==== OLLAMA ====
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

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ";"]
        )

    # ---------- PARSER ----------
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

    # ---------- PERMANENCE (Structure maintained) ----------
    def rate_limit_sleep(self, model_id: str):
        return

    def _partial_path(self):
        Path("output").mkdir(exist_ok=True)
        return Path("output") / f"partial_{self.output_file}"

    def load_partial_if_exists(self):
        path = self._partial_path()
        if path.exists():
            logger.info("🔁 Melanjutkan dari file partial")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.qa_pairs = data.get("qa_pairs", [])

    def save_partial(self, source_file: str, model_id: str):
        with self._partial_path().open("w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "source": source_file,
                    "model": model_id,
                    "total_qa_pairs": len(self.qa_pairs),
                    "last_saved": datetime.datetime.now().isoformat(),
                    "status": "PARTIAL"
                },
                "qa_pairs": self.qa_pairs
            }, f, indent=2, ensure_ascii=False)

    # ---------- MAIN PROCESS ----------
    def process_json(self, json_path: Path, model_id: str):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            st.error("❌ JSON harus berupa LIST")
            return

        self.load_partial_if_exists()

        start_time = datetime.datetime.now().isoformat()

        progress = st.progress(0.0)
        status = st.empty()

        for i, entry in enumerate(data):

            kategori = entry.get("metadata", {}).get("kategori", "").strip()
            text = entry.get("page_content", "").strip()

            if not text:
                continue

            docs = self.text_splitter.create_documents([text])
            chunk_qa = 0

            for idx, doc in enumerate(docs):

                output = self.ai_manager.generate(
                    doc.page_content,
                    kategori,
                    model_id
                )

                if not output:
                    continue

                qa = self.parse_qa(output)
                if qa:
                    self.qa_pairs.extend(qa)
                    chunk_qa += len(qa)
                    self.save_partial(json_path.name, model_id)
                else:
                    logger.info(f"ℹ️ Chunk {idx+1} tanpa QA")

            progress.progress((i + 1) / len(data))
            status.text(f"Memproses chunk {i+1}/{len(data)} — kategori: {kategori}")

        end_time = datetime.datetime.now().isoformat()

        # ---------- FINAL SAVE ----------
        output_path = Path("output") / self.output_file
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "source": json_path.name,
                    "model": model_id,
                    "total_qa_pairs": len(self.qa_pairs),
                    "start_generate_at": start_time,
                    "end_generate_at": end_time,
                    "status_generate": "FINAL"
                },
                "qa_pairs": self.qa_pairs
            }, f, indent=2, ensure_ascii=False)

        st.success(f"✅ Dataset selesai: {output_path}")



# =========================
# STREAMLIT APP
# =========================
def main():
    st.set_page_config("Synthetic Q&A Generator", layout="wide")
    st.title("📘 Synthetic Q&A Dataset Generator (PERTOR JSON)")

    load_dotenv()

    provider = st.sidebar.selectbox("AI Provider", ["ollama"])

    data_dir = Path("data")
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        st.error("❌ Folder data/ kosong")
        return

    json_file = st.selectbox("Pilih File JSON", json_files, format_func=lambda x: x.name)

    ai = AIManager(provider)
    models = ai.get_available_models()

    if not models:
        st.error("❌ Tidak ada model Ollama. Jalankan: ollama pull llama3.1")
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