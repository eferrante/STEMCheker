import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

COMETAPI_BASE_URL = "https://api.cometapi.com/v1"
MODELS = {
    "Claude Opus 4.7  — $3 / 1M tokens": "claude-opus-4-7",
    "Claude Sonnet 4.6 — $2.4 / 1M tokens (cheaper)": "claude-sonnet-4-6",
}
DEFAULT_MODEL_LABEL = "Claude Opus 4.7  — $3 / 1M tokens"

PROMPT_PLACEHOLDER = "<PROMPT ENTERED IN THE INTERFACE>"
ANSWER_PLACEHOLDER = "<GOLDEN ANSWER ENTERED IN THE INTERFACE>"

DEFAULT_TEMPLATE = Path(__file__).parent / "default_system_prompt.txt"


def get_api_key() -> str:
    # Streamlit Cloud secrets take priority, then .env / shell env
    try:
        return st.secrets["COMETAPI_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("COMETAPI_KEY", "")


@st.cache_data
def load_default_template() -> str:
    return DEFAULT_TEMPLATE.read_text(encoding="utf-8")


def stream_content(stream):
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def run_review(client: OpenAI, model: str, prompt_template: str, task_prompt: str, golden_answer: str) -> None:
    full_message = prompt_template.replace(PROMPT_PLACEHOLDER, task_prompt).replace(
        ANSWER_PLACEHOLDER, golden_answer
    )
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_message}],
        stream=True,
    )
    st.write_stream(stream_content(stream))


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="STEM Annotation QA",
    page_icon="🔬",
    layout="wide",
)

api_key = get_api_key()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    model_label = st.selectbox("Model", list(MODELS.keys()), index=list(MODELS.keys()).index(DEFAULT_MODEL_LABEL))
    model = MODELS[model_label]
    if not api_key:
        st.error("COMETAPI_KEY secret is not configured.")

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🔬 STEM Annotation Quality Checker")
st.caption("Review annotation tasks against project guidelines using Claude via CometAPI.")

with st.expander("System Prompt (editable)", expanded=False):
    prompt_template = st.text_area(
        "template",
        value=load_default_template(),
        height=420,
        label_visibility="collapsed",
        help=f"Uses {PROMPT_PLACEHOLDER!r} and {ANSWER_PLACEHOLDER!r} as substitution markers.",
    )

st.divider()

task_prompt = st.text_area(
    "Task Prompt",
    height=250,
    placeholder="Paste the tasker's question here...",
)

golden_answer = st.text_area(
    "Golden Answer",
    height=120,
    placeholder="Paste the golden answer here...",
)

check_btn = st.button(
    "Check Quality",
    type="primary",
    disabled=not api_key,
    use_container_width=True,
)

if not api_key:
    st.info("COMETAPI_KEY is not set. Configure it as a secret in Streamlit Cloud or in the local .env file.")

if check_btn:
    if not task_prompt.strip():
        st.warning("Please enter the task prompt.")
    elif not golden_answer.strip():
        st.warning("Please enter the golden answer.")
    elif PROMPT_PLACEHOLDER not in prompt_template or ANSWER_PLACEHOLDER not in prompt_template:
        st.error(
            f"The system prompt template must contain both placeholders:\n"
            f"- `{PROMPT_PLACEHOLDER}`\n"
            f"- `{ANSWER_PLACEHOLDER}`"
        )
    else:
        client = OpenAI(api_key=api_key, base_url=COMETAPI_BASE_URL)
        st.divider()
        st.subheader("Review")
        try:
            run_review(client, model, prompt_template, task_prompt, golden_answer)
        except Exception as exc:
            st.error(f"API error: {exc}")
