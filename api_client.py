import time
import json
import re
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (
    API_KEY,
    OPENAI_BASE_URL,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_ONLY,
    MODEL_OVERRIDES,
    MAX_TOKENS,
    TEMPERATURE,
)

_custom_api_key = None
_custom_base_url = None


def set_custom_api(api_key: str = None, base_url: str = None):
    global _custom_api_key, _custom_base_url
    _custom_api_key = api_key
    _custom_base_url = base_url


def get_api_key():
    return _custom_api_key or API_KEY


def _openai_base():
    return _custom_base_url or OPENAI_BASE_URL


def _anthropic_base():
    return _custom_base_url or ANTHROPIC_BASE_URL


def _headers_openai() -> dict:
    return {"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"}


def _headers_anthropic() -> dict:
    return {
        "x-api-key": get_api_key(),
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }


def _get_temperature(model_id: str) -> float:
    overrides = MODEL_OVERRIDES.get(model_id, {})
    return overrides.get("temperature", TEMPERATURE)


_THINKING_STARTS = [
    "**分析", "**Analysis", "**分析请求", "**Reviewing",
    "**梳理", "**理解题", "**解读", "**思考过程",
    "**Let me think", "**Let me analyze", "**My analysis",
    "**Reasoning", "**Step", "**步骤",
]

_DEBATE_MARKERS = [
    "各位", "我方认为", "对方辩友",
    "我方坚持", "我方立场",
    "首先", "综上所述", "总之", "结论", "因此，",
]

_STRAY_STARTS = [
    "Wait,", "Actually,", "Hmm,", "Let me",
    "Note:", "注意：", "等一下", "The user",
    "I should", "I need", "让我", "我需要",
]

_ENGLISH_THINKING_PATTERNS = [
    re.compile(r"^The (debate )?topic is", re.IGNORECASE),
    re.compile(r"^Positions so far", re.IGNORECASE),
    re.compile(r"^This is round \d", re.IGNORECASE),
    re.compile(r"^I need to (present|argue|make|focus|provide)", re.IGNORECASE),
    re.compile(r"^What (new angle|key point|argument|perspective)", re.IGNORECASE),
    re.compile(r"^(Some ideas|Key angles|Key points|Main points) ", re.IGNORECASE),
    re.compile(r"^\d+\. (Compare|Focus|Look|Challenge|Bring|Point|Emphasize)", re.IGNORECASE),
    re.compile(r"^New angle:", re.IGNORECASE),
    re.compile(r"^I think a good", re.IGNORECASE),
    re.compile(r"^Let me (think|analyze|consider|review|reconsider)", re.IGNORECASE),
    re.compile(r"^\*{0,2}Hmm", re.IGNORECASE),
    re.compile(r"^Almost all", re.IGNORECASE),
]


def strip_thinking(text: str) -> str:
    if not text:
        return text

    # Remove style tags
    text = re.sub(r"", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL)

    # Detect and strip leading English thinking paragraphs
    # If the text starts with English analysis and later transitions to Chinese debate content
    lines = text.split("\n")
    first_significant_chinese = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        chinese = sum(1 for c in stripped if "\u4e00" <= c <= "\u9fff")
        total = len(stripped)
        if total > 10 and chinese > total * 0.4:
            # Check this line contains a debate marker
            if any(m in stripped for m in _DEBATE_MARKERS):
                first_significant_chinese = i
                break

    if first_significant_chinese > 0:
        # Check if the preceding lines are predominantly English thinking
        preceding = "\n".join(lines[:first_significant_chinese])
        english_chars = sum(1 for c in preceding if c.isascii() and c.isalpha())
        chinese_chars = sum(1 for c in preceding if "\u4e00" <= c <= "\u9fff")
        if english_chars > chinese_chars * 2 and english_chars > 20:
            text = "\n".join(lines[first_significant_chinese:])

    # Now do line-by-line processing
    lines = text.split("\n")
    clean = []
    in_thinking = False
    for line in lines:
        stripped = line.strip()

        if any(stripped.startswith(p) for p in _THINKING_STARTS):
            in_thinking = True
            continue

        if in_thinking:
            if any(stripped.startswith(m) for m in _DEBATE_MARKERS):
                in_thinking = False
                clean.append(line)
            elif stripped.startswith("**") and not stripped.startswith("**分析"):
                in_thinking = False
                clean.append(line)
            elif stripped.startswith(("1.", "2.", "3.", "-")):
                in_thinking = False
                clean.append(line)
            continue

        # Skip English thinking lines
        if any(pat.match(stripped) for pat in _ENGLISH_THINKING_PATTERNS):
            continue

        if any(stripped.startswith(p) for p in _STRAY_STARTS):
            continue

        clean.append(line)

    text = "\n".join(clean)

    # Find the first debate-like content marker
    marker_positions = [(text.find(m), m) for m in _DEBATE_MARKERS if text.find(m) > 0]
    if marker_positions:
        best = min(pos for pos, _ in marker_positions)
        before = text[:best].rstrip()
        chinese = sum(1 for c in before if "\u4e00" <= c <= "\u9fff")
        if chinese > len(before) * 0.3 and len(text[best:]) >= 20:
            text = text[best:]

    return text.strip()


def call_model(model_id: str, messages: list[dict], system_prompt: str = "") -> str:
    if model_id in ANTHROPIC_ONLY:
        raw = _call_anthropic(model_id, messages, system_prompt)
    else:
        raw = _call_openai(model_id, messages, system_prompt)
    return strip_thinking(raw)


def call_models_parallel(calls: list[tuple]) -> list:
    """Call multiple models in parallel. Each call is (model_id, messages, system_prompt)."""
    results = [None] * len(calls)
    with ThreadPoolExecutor(max_workers=min(len(calls), 5)) as executor:
        futures = {}
        for i, (model_id, messages, system_prompt) in enumerate(calls):
            future = executor.submit(call_model, model_id, messages, system_prompt)
            futures[future] = i
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"[调用失败: {e}]"
    return results


def _retry(fn, description: str = "", max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 10 * attempt
                print(f"[retry] 429 rate limit, waiting {wait}s ({attempt}/{max_retries})")
                time.sleep(wait)
            elif e.response.status_code >= 500:
                wait = 5 * attempt
                print(f"[retry] {e.response.status_code}, waiting {wait}s ({attempt}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            wait = 5 * attempt
            print(f"[retry] {type(e).__name__}, waiting {wait}s ({attempt}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"{description} failed after {max_retries} retries")


def _call_openai(model_id: str, messages: list[dict], system_prompt: str) -> str:
    def _do():
        temp = _get_temperature(model_id)
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": temp,
        }
        if system_prompt:
            payload["messages"] = [
                {"role": "system", "content": system_prompt}
            ] + messages
        resp = httpx.post(
            f"{_openai_base()}/chat/completions",
            headers=_headers_openai(),
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or ""
        return content.strip()

    return _retry(_do, f"OpenAI {model_id}")


def _call_anthropic(model_id: str, messages: list[dict], system_prompt: str) -> str:
    def _do():
        anthropic_messages = []
        for m in messages:
            if m["role"] in ("user", "assistant"):
                anthropic_messages.append(
                    {"role": m["role"], "content": m["content"]}
                )
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": "..."})
        if len(anthropic_messages) % 2 == 0:
            anthropic_messages.insert(0, {"role": "user", "content": "..."})

        payload = {
            "model": model_id,
            "messages": anthropic_messages,
            "max_tokens": MAX_TOKENS,
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = httpx.post(
            f"{_anthropic_base()}/messages",
            headers=_headers_anthropic(),
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"].strip()
        if data.get("content"):
            last = data["content"][-1]
            return (last.get("text") or "").strip()
        return ""

    return _retry(_do, f"Anthropic {model_id}")


def parse_json_response(text: str) -> dict:
    json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
