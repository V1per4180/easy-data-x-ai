import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from langchain.chat_models import init_chat_model

MODEL_NAME = "deepseek-flash"


def run_demo(llm):
    response = llm.invoke(
        [
            ("system", "你是一个简洁高效的技术助手。用中文回答，尽量用一两句话。"),
            ("user", "什么是 RAG？"),
        ]
    )
    print(response.content)
    return response


def main(model_factory=init_chat_model) -> int:
    # Key 边界：模型初始化前失败，避免产生任何外部调用。
    try:
        Config.require_api_key("OPENAI_API_KEY")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    llm = model_factory(
        MODEL_NAME,
        model_provider="openai",
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_BASE_URL,
        extra_body={"thinking": {"type": "disabled"}},
    )
    run_demo(llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
