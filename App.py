import sys
import os
import json
from pydantic import BaseModel, Field
import streamlit as st
from dotenv import load_dotenv

# =========================================================================
# 1. ENVIRONMENT SAFEGUARDS & CONFIGURATION LAYER
# =========================================================================
# Python 3.9-3.11 persistent binary override for native SQLite engine runtimes
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "sports_facts.json")

def validate_and_warmup_environment():
    """Ensures environment configurations match production mandates and hooks local data directories."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        raise ValueError("[CRITICAL CONFIGURATION ERROR]: Environment failed to detect an active 'OPENAI_API_KEY'. Check your config alignment.")
    
    # Generate mock database file locally if missing to guarantee operational continuity
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        sample_data = [
            {"fact": "The modern Olympic Games were first held in Athens, Greece, in 1896, featuring 241 athletes.", "sport": "Cricket"},
            {"fact": "Cricket made its only appearance at the modern Olympic Games during the 1800 Summer Olympics in Paris.", "sport": "Cricket"},
            {"fact": "In FIFA World Cup history, Brazil has won the tournament the most times, securing 5 titles.", "sport": "Football"},
            {"fact": "Prakash Padukone was the first Indian to win the prestigious All England Open Badminton Championships in 1980.", "sport": "Badminton"}
        ]
        with open(DATA_PATH, "w") as f:
            json.dump(sample_data, f, indent=2)

# =========================================================================
# 2. PERSISTENT VECTOR STORAGE MODULE (RAG FACTUAL VALIDATION LAYER)
# =========================================================================
def get_chroma_client():
    """Returns a thread-safe persistent vector database connection pointing directly to disk boundaries."""
    return chromadb.PersistentClient(path=DB_DIR)

def setup_and_populate_db():
    """
    Validates, vectors, and persists baseline historical milestones into ChromaDB collections.
    Employs short-circuit logical protection to eliminate systemic document duplication profiles.
    """
    client = get_chroma_client()
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="sports_history",
        embedding_function=embedding_fn
    )

    # SHORT-CIRCUIT PROTECTION: Detect if collection nodes have already been compiled
    if collection.count() > 0:
        return collection

    with open(DATA_PATH, "r") as f:
        facts_list = json.load(f)

    documents = []
    metadata_list = []
    ids = []

    for idx, item in enumerate(facts_list):
        documents.append(item["fact"])
        metadata_list.append({"sport": item["sport"]})
        ids.append(f"fact_{idx}")

    collection.add(
        documents=documents,
        metadatas=metadata_list,
        ids=ids
    )
    return collection

def query_historic_facts(sport: str, query_text: str, n_results: int = 2) -> list:
    """Executes semantic mathematical exploration queries utilizing metadata categorization vectors."""
    client = get_chroma_client()
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="sports_history",
        embedding_function=embedding_fn
    )
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"sport": sport}
    )
    return results.get("documents", [[]])[0]

# =========================================================================
# 3. INTERNET EXTRACTION AGENT (DYNAMIC REAL-TIME DATA LAYER)
# =========================================================================
def get_live_news_context(sport_name: str) -> str:
    """
    Scours public internet networks on-demand to fetch real-time updates.
    Contains complete connection insulation matrices to maintain overall system uptime.
    """
    # System Constraint Enforcement: Temporal Target Alignment explicitly set to operational year 2026
    search_query = f"{sport_name} latest tournament results championship winners news 2026"
    retrieved_texts = []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
            for index, r in enumerate(results, start=1):
                title = r.get("title", "UNTITLED_NODE")
                snippet = r.get("body", "EMPTY_SNIPPET_STREAM")
                retrieved_texts.append(f"Web Source {index}: {title}\nPayload content: {snippet}")
    except Exception as e:
        # FAULT TOLERANCE GRACEFUL DEGRADATION BLOCK:
        return f"System Warning: Network context execution bypassed due to: {e}. Pipeline fell back smoothly to Isolated Local Data Operations."

    return "\n\n".join(retrieved_texts)

# =========================================================================
# 4. STRATEGY 2: NATIVE STRUCTURAL JSON FORMAT CONTROL SCHEMAS & AGENT
# =========================================================================
class QuestionOptions(BaseModel):
    A: str = Field(description="Option A text response value.")
    B: str = Field(description="Option B text response value.")
    C: str = Field(description="Option C text response value.")
    D: str = Field(description="Option D text response value.")

class QuizItem(BaseModel):
    question: str = Field(description="The actual evaluation question text generated from ground truth context.")
    options: QuestionOptions = Field(description="Exactly four distinct multiple-choice option parameters.")
    correct_answer: str = Field(description="The single characters code verifying the true option choice: A, B, C, or D.")
    explanation: str = Field(description="Factual analytical reasoning directly citing statements located inside context fields.")

class QuizPayloadSchema(BaseModel):
    quizzes: list[QuizItem] = Field(description="An explicit collection list of uniquely compiled quiz evaluation modules.")

def compile_quiz_data(sport: str, difficulty: str) -> tuple[dict, str]:
    """
    Central Core Orchestration Sequence mapping to OpenAI's contemporary Structured Outputs format.
    Ensures absolute structure reliability by forcing constrained decoding.
    """
    db_query = f"{sport} legendary historical championships rules milestones"
    db_matches = query_historic_facts(sport=sport, query_text=db_query, n_results=2)
    db_context = "\n".join(db_matches) if db_matches else "No offline background items recorded."
    
    web_context = get_live_news_context(sport)
    unified_context = f"=== GROUND TRUTH HISTORICAL RECORDS ===\n{db_context}\n\n=== RUNTIME INTERNET SCROLL ===\n{web_context}"

    client = OpenAI(api_key=OPENAI_API_KEY)

    advanced_system_instruction = (
        "You are an expert sports quiz programmatic API engine. Output a strict JSON object matching the requested schema layout.\n"
        "Rely strictly on the provided Context. Avoid hallucinations. Do not use facts not found in the Context below.\n"
        "If facts are scarce, make do with what you have, but keep details completely accurate to the text context.\n\n"
        f"CONTEXT DETAILS:\n{unified_context}"
    )

    advanced_user_prompt = (
        f"Generate exactly 3 unique multiple-choice quiz entries target-tuned for the sport: {sport}. "
        f"Difficulty target configuration value: {difficulty}."
    )

    # Use the parsing helper engine ensuring structural layout compliance
    response = client.beta.chat.completions.parse(
        model="gpt-3.5-turbo",
        response_format=QuizPayloadSchema,
        messages=[
            {"role": "system", "content": advanced_system_instruction},
            {"role": "user", "content": advanced_user_prompt}
        ],
        temperature=0.1
    )

    parsed_response = response.choices[0].message.parsed
    if parsed_response is None:
        raise ValueError("The LLM refused or generated a schema payload configuration mismatch exception.")
        
    return parsed_response.model_dump(), unified_context

# =========================================================================
# 5. STREAMLIT UI PRESENTATION ARCHITECTURE
# =========================================================================
st.set_page_config(page_title="AI Sports Quiz Framework", page_icon="🏆", layout="centered")

try:
    validate_and_warmup_environment()
except ValueError as e:
    st.error(str(e))
    st.stop()

@st.cache_resource
def seed_system_data_layer():
    """Warms up local file access points and seeds vector space metrics once per lifecycle session."""
    setup_and_populate_db()

seed_system_data_layer()

st.title("🏆 AI Sports Quiz Verification Engine")
st.write("Factually anchored quiz automation framework incorporating local semantic vector repositories and live runtime web scrapers.")

st.sidebar.header("Agent Infrastructure Directives")
sport_input = st.sidebar.selectbox("Select Core Discipline Focus", ["Cricket", "Football", "Badminton"])
difficulty_input = st.sidebar.select_slider("Select Target Evaluation Range", options=["Easy", "Medium", "Hard"])

# Persistent State Tracking Setup
if "persistent_output_frame" not in st.session_state:
    st.session_state.persistent_output_frame = None
if "persistent_context_frame" not in st.session_state:
    st.session_state.persistent_context_frame = None
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if st.sidebar.button("Trigger Generation Sequence", use_container_width=True):
    with st.spinner("Extracting historical data coordinates and scanning public networks..."):
        try:
            quiz_data, context_string = compile_quiz_data(sport_input, difficulty_input)
            st.session_state.persistent_output_frame = quiz_data
            st.session_state.persistent_context_frame = context_string
            st.session_state.form_submitted = False
            st.success("RAG synthesis context compilation finalized successfully.")
        except Exception as e:
            st.error(f"Critical workflow interruption: {e}")

# Render Execution Layout Module Frames
if st.session_state.persistent_output_frame:
    quiz_records = st.session_state.persistent_output_frame.get("quizzes", [])
    
    st.subheader(f"Active Interactive Evaluation Manifest: {sport_input} ({difficulty_input})")
    
    with st.form(key="quiz_interactive_form"):
        for index, item in enumerate(quiz_records):
            st.markdown(f"#### **Q{index + 1}: {item['question']}**")
            opts = item["options"]
            option_choices = [
                f"A: {opts['A']}",
                f"B: {opts['B']}",
                f"C: {opts['C']}",
                f"D: {opts['D']}"
            ]
            
            st.radio(
                "Options:",
                options=option_choices,
                key=f"user_selection_{index}",
                label_visibility="collapsed"
            )
        
        submit_btn = st.form_submit_button("Verify Performance Scorecard")
        
    if submit_btn or st.session_state.form_submitted:
        st.session_state.form_submitted = True
        correct_tally = 0
        
        st.markdown("### 📊 Verification Analysis Summary")
        for index, item in enumerate(quiz_records):
            raw_selection = st.session_state.get(f"user_selection_{index}", "A:")
            selected_letter = raw_selection.split(":")[0].strip()
            target_answer = item["correct_answer"].strip()
            
            st.markdown(f"**Question {index + 1}:** {item['question']}")
            if selected_letter == target_answer:
                correct_tally += 1
                st.success(f"✔️ Score Matching Confirmed! Selected: {selected_letter}")
            else:
                st.error(f"❌ Structural Discrepancy Found. Selected: {selected_letter} | System Expected: {target_answer}")
                
            st.info(f"💡 **Factual Background:** {item['explanation']}")
            st.markdown("---")
            
        accuracy_rating = (correct_tally / len(quiz_records)) * 100
        st.metric("Total Factual Accuracy Index", f"{accuracy_rating:.1f}%", f"{correct_tally}/{len(quiz_records)} Target Matches")

    with st.expander("🔍 Enterprise Audit Log: Inspect Ground Truth RAG Ingestion Matrix"):
        st.write("The underlying facts below were dynamically captured and inserted into the system instructions block:")
        st.code(st.session_state.persistent_context_frame, language="markdown")
