import json
import os
import urllib.error
import urllib.request


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:0.6b",
)


def call_mock_llm(prompt: str) -> str:
    """临时模拟大模型回复，保证项目流程可以先运行。"""
    del prompt
    return "你好"


def call_ollama(prompt: str) -> str:
    """通过本地 Ollama API 调用大模型。"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            "无法连接 Ollama，请确认 Ollama 已安装并正在运行。"
        ) from error

    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError(f"Ollama 返回内容异常：{result}")

    return answer


def call_llm(prompt: str) -> str:
    """
    LLM_MODE=mock：使用临时模拟接口。
    LLM_MODE=ollama：调用本地 Ollama。
    """
    mode = os.getenv("LLM_MODE", "mock").lower()

    if mode == "mock":
        return call_mock_llm(prompt)

    if mode == "ollama":
        return call_ollama(prompt)

    raise ValueError(
        f"不支持的 LLM_MODE：{mode}，只能填写 mock 或 ollama。"
    )


if __name__ == "__main__":
    reply = call_llm("请只回复两个汉字：你好")
    print("LLM 模式：", os.getenv("LLM_MODE", "mock"))
    print("LLM 返回：", reply)