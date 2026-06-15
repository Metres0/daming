import os
import json
from pathlib import Path

AUTH_FILE = Path.home() / ".local" / "share" / "opencode" / "auth.json"


def _load_api_key() -> str:
    env_key = os.environ.get("OPENCODE_GO_API_KEY")
    if env_key:
        return env_key
    if AUTH_FILE.exists():
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        return data.get("opencode-go", {}).get("key", "")
    return ""


API_KEY = _load_api_key()

OPENAI_BASE_URL = "https://opencode.ai/zen/go/v1"
ANTHROPIC_BASE_URL = "https://opencode.ai/zen/go/v1"

ANTHROPIC_ONLY = {"qwen3.7-max"}

MODEL_OVERRIDES = {
    "kimi-k2.7-code": {"temperature": 1},
    "mimo-v2.5": {"temperature": 1},
    "mimo-v2.5-pro": {"temperature": 1},
}

ALL_MODELS = {
    "glm-5.1": {
        "name": "GLM-5.1",
        "emoji": "\U0001f9e0",
        "color": "#00d4aa",
    },
    "glm-5": {
        "name": "GLM-5",
        "emoji": "\U0001f9e0",
        "color": "#00b894",
    },
    "kimi-k2.7-code": {
        "name": "Kimi K2.7 Code",
        "emoji": "\U0001f319",
        "color": "#e056a0",
    },
    "kimi-k2.6": {
        "name": "Kimi K2.6",
        "emoji": "\U0001f319",
        "color": "#be2edd",
    },
    "deepseek-v4-pro": {
        "name": "DeepSeek V4 Pro",
        "emoji": "\U0001f50d",
        "color": "#0984e3",
    },
    "deepseek-v4-flash": {
        "name": "DeepSeek V4 Flash",
        "emoji": "\u26a1",
        "color": "#74b9ff",
    },
    "mimo-v2.5": {
        "name": "MiMo V2.5",
        "emoji": "\U0001f431",
        "color": "#fdcb6e",
    },
    "mimo-v2.5-pro": {
        "name": "MiMo V2.5 Pro",
        "emoji": "\U0001f431",
        "color": "#f9ca24",
    },
    "minimax-m3": {
        "name": "MiniMax M3",
        "emoji": "\U0001f3b5",
        "color": "#00cec9",
    },
    "minimax-m2.7": {
        "name": "MiniMax M2.7",
        "emoji": "\U0001f3b5",
        "color": "#55efc4",
    },
    "qwen3.7-max": {
        "name": "Qwen3.7 Max",
        "emoji": "\U0001f432",
        "color": "#e17055",
    },
    "qwen3.7-plus": {
        "name": "Qwen3.7 Plus",
        "emoji": "\U0001f432",
        "color": "#d63031",
    },
    "qwen3.6-plus": {
        "name": "Qwen3.6 Plus",
        "emoji": "\U0001f432",
        "color": "#b2bec3",
    },
}

DEFAULT_DEBATE_MODELS = [
    "glm-5.1",
    "kimi-k2.6",
    "deepseek-v4-flash",
    "qwen3.7-plus",
    "minimax-m3",
]

DEFAULT_GROUP_A = ["glm-5.1", "kimi-k2.6", "deepseek-v4-flash"]
DEFAULT_GROUP_B = ["qwen3.7-plus", "minimax-m3", "qwen3.6-plus"]

MAX_TOKENS = 512
TEMPERATURE = 0.8

SCORING_SYSTEM_PROMPT = """你是一位专业的辩论评委。请对以下辩论发言进行评分。

评分标准（每项1-10分）：
1. 论点力度：论点是否有力、有深度
2. 逻辑严密：推理是否严密、有无逻辑漏洞
3. 说服力：整体是否具有说服力
4. 表达质量：语言是否清晰、有条理

请以JSON格式输出评分，格式如下：
{"论点力度": 8, "逻辑严密": 7, "说服力": 8, "表达质量": 9, "总评": "一句话总结评价"}

只输出JSON，不要输出其他内容。不要输出思考过程。"""

MUTUAL_SCORING_PROMPT = """你是一位辩论参与者，同时也是评委。请从你自己的观点立场出发，对以下其他辩手的发言进行评分。

你的身份：{identity}
评分标准（每项1-10分）：
1. 论点力度：论点是否有力、有深度
2. 逻辑严密：推理是否严密
3. 说服力：整体是否具有说服力
4. 表达质量：语言是否清晰

请以JSON格式输出评分，格式如下：
{{"论点力度": 8, "逻辑严密": 7, "说服力": 8, "表达质量": 9, "总评": "一句话总结"}}

只输出JSON，不要输出其他内容。不要输出思考过程。"""