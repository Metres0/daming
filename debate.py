import time
import json
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from config import ALL_MODELS, DEFAULT_DEBATE_MODELS, DEFAULT_GROUP_A, DEFAULT_GROUP_B, SCORING_SYSTEM_PROMPT, MUTUAL_SCORING_PROMPT
from api_client import call_model, call_models_parallel, parse_json_response


class DebateManager:
    def __init__(self):
        self.sessions = {}
        self.events = {}
        self.lock = threading.Lock()

    def _add_event(self, session_id, event):
        with self.lock:
            if session_id not in self.events:
                self.events[session_id] = []
            self.events[session_id].append(event)

    def _init_session(self, session_id, mode, topic):
        self.sessions[session_id] = {
            "id": session_id,
            "mode": mode,
            "topic": topic,
            "history": [],
            "scores": [],
            "summary": None,
            "status": "running",
            "created_at": datetime.now().isoformat(),
        }
        self.events[session_id] = []

    def get_next_event(self, session_id):
        with self.lock:
            events = self.events.get(session_id, [])
            if events:
                return events.pop(0)
        return None

    def is_finished(self, session_id):
        session = self.sessions.get(session_id)
        if not session:
            return True
        return session.get("status") in ("finished", "error")

    def get_history(self, session_id):
        session = self.sessions.get(session_id)
        if not session:
            return None
        return {
            "id": session["id"],
            "mode": session["mode"],
            "topic": session["topic"],
            "history": session["history"],
            "scores": session.get("scores", []),
            "status": session["status"],
        }

    def run_roundtable(self, session_id, topic, models, rounds):
        try:
            self._init_session(session_id, "roundtable", topic)
            self._add_event(session_id, {
                "type": "start",
                "mode": "roundtable",
                "topic": topic,
                "models": models,
                "rounds": rounds,
            })

            history = []
            for round_num in range(1, rounds + 1):
                self._add_event(session_id, {
                    "type": "round_start",
                    "round": round_num,
                    "total_rounds": rounds,
                })

                # Send all thinking events first
                for model_id in models:
                    self._add_event(session_id, {
                        "type": "thinking",
                        "model": model_id,
                        "model_name": ALL_MODELS[model_id]["name"],
                        "round": round_num,
                    })

                # Call all models in parallel within the round
                calls = []
                for model_id in models:
                    role = f"第{models.index(model_id)+1}位发言者"
                    system = _build_system_prompt(role, topic, round_num=round_num)
                    messages = _build_messages(topic, history, round_num)
                    calls.append((model_id, messages, system))

                results = call_models_parallel(calls)

                for i, model_id in enumerate(models):
                    response = results[i]

                    entry = {
                        "model": model_id,
                        "model_name": ALL_MODELS[model_id]["name"],
                        "content": response,
                        "round": round_num,
                        "side": "",
                        "timestamp": datetime.now().isoformat(),
                    }
                    history.append(entry)
                    self.sessions[session_id]["history"] = history

                    self._add_event(session_id, {"type": "speech", **entry})

            self._add_event(session_id, {
                "type": "debate_done",
                "history": history,
            })
            self.sessions[session_id]["status"] = "finished"
            self._add_event(session_id, {"type": "done"})

        except Exception as e:
            self.sessions.setdefault(session_id, {})["status"] = "error"
            self._add_event(session_id, {"type": "error", "message": str(e)})
            self._add_event(session_id, {"type": "done"})

    def run_group(self, session_id, topic, group_a, group_b, rounds):
        try:
            self._init_session(session_id, "group", topic)
            self._add_event(session_id, {
                "type": "start",
                "mode": "group",
                "topic": topic,
                "group_a": group_a,
                "group_b": group_b,
                "rounds": rounds,
            })

            history = []
            for round_num in range(1, rounds + 1):
                self._add_event(session_id, {
                    "type": "round_start",
                    "round": round_num,
                    "total_rounds": rounds,
                })

                for model_id in group_a:
                    self._add_event(session_id, {
                        "type": "thinking",
                        "model": model_id,
                        "model_name": ALL_MODELS[model_id]["name"],
                        "round": round_num,
                        "side": "正方",
                    })

                system_a = _build_system_prompt("正方辩手", topic, "正方", round_num)
                messages_a = _build_messages(topic, history, round_num, "正方")
                calls_a = [(mid, messages_a, system_a) for mid in group_a]
                results_a = call_models_parallel(calls_a)

                for i, model_id in enumerate(group_a):
                    response = results_a[i]
                    entry = {
                        "model": model_id,
                        "model_name": ALL_MODELS[model_id]["name"],
                        "content": response,
                        "round": round_num,
                        "side": "正方",
                        "timestamp": datetime.now().isoformat(),
                    }
                    history.append(entry)
                    self.sessions[session_id]["history"] = history
                    self._add_event(session_id, {"type": "speech", **entry})

                for model_id in group_b:
                    self._add_event(session_id, {
                        "type": "thinking",
                        "model": model_id,
                        "model_name": ALL_MODELS[model_id]["name"],
                        "round": round_num,
                        "side": "反方",
                    })
                    system_b = _build_system_prompt("反方辩手", topic, "反方", round_num)
                messages_b = _build_messages(topic, history, round_num, "反方")
                calls_b = [(mid, messages_b, system_b) for mid in group_b]
                results_b = call_models_parallel(calls_b)

                for i, model_id in enumerate(group_b):
                    response = results_b[i]
                    entry = {
                        "model": model_id,
                        "model_name": ALL_MODELS[model_id]["name"],
                        "content": response,
                        "round": round_num,
                        "side": "反方",
                        "timestamp": datetime.now().isoformat(),
                    }
                    history.append(entry)
                    self.sessions[session_id]["history"] = history
                    self._add_event(session_id, {"type": "speech", **entry})

            self._add_event(session_id, {
                "type": "debate_done",
                "history": history,
            })
            self.sessions[session_id]["status"] = "finished"
            self._add_event(session_id, {"type": "done"})

        except Exception as e:
            self.sessions.setdefault(session_id, {})["status"] = "error"
            self._add_event(session_id, {"type": "error", "message": str(e)})
            self._add_event(session_id, {"type": "done"})

    def generate_summary(self, session_id, model_id, summary_type="auto"):
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "会话不存在"}

        history = session.get("history", [])
        topic = session.get("topic", "")
        mode = session.get("mode", "roundtable")

        all_content = _format_history(history)

        if mode == "group":
            system = (
                "你是一位辩论裁判。请对以下分组辩论进行客观评判，包括：正方主要论点、"
                "反方主要论点、争论焦点、以及你认为哪方论证更有说服力（需给出理由）。"
                "控制在400字以内，用中文回答。"
            )
            prompt = f"辩论话题：{topic}\n\n辩论记录：\n{all_content}\n\n请给出裁判总结。"
        else:
            system = (
                "你是一位辩论总结专家。请对以下辩论内容进行客观、全面的总结，"
                "包括各方主要观点、争论焦点和最终结论。控制在300字以内，用中文回答。"
            )
            prompt = f"辩论话题：{topic}\n\n辩论记录：\n{all_content}\n\n请给出总结。"

        messages = [{"role": "user", "content": prompt}]

        try:
            summary = call_model(model_id, messages, system)
        except Exception as e:
            summary = f"[生成总结失败: {e}]"

        result = {
            "model": model_id,
            "model_name": ALL_MODELS.get(model_id, {}).get("name", model_id),
            "content": summary,
            "summary_type": summary_type,
        }

        session["summary"] = result
        return result

    def score_speeches(self, session_id, scorer_model_id, speech_indices=None):
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "会话不存在"}

        history = session.get("history", [])
        topic = session.get("topic", "")

        speeches_to_score = []
        for i, entry in enumerate(history):
            if speech_indices is None or i in speech_indices:
                speeches_to_score.append((i, entry))

        calls = []
        for idx, entry in speeches_to_score:
            speech_text = f"[{entry.get('model_name', entry.get('model'))}"
            side = entry.get("side", "")
            if side:
                speech_text += f" ({side})"
            speech_text += f"]: {entry['content']}"
            prompt = f"辩论话题：{topic}\n\n请对以下发言进行评分：\n\n{speech_text}"
            calls.append((scorer_model_id, [{"role": "user", "content": prompt}], SCORING_SYSTEM_PROMPT))

        raw_results = call_models_parallel(calls)

        scores = []
        for i, (idx, entry) in enumerate(speeches_to_score):
            raw = raw_results[i]
            try:
                parsed = parse_json_response(raw)
                parsed["raw_response"] = raw
            except Exception:
                parsed = {"raw": raw}

            score_entry = {
                "speech_index": idx,
                "model": entry.get("model"),
                "model_name": entry.get("model_name", entry.get("model")),
                "side": entry.get("side", ""),
                "scorer_model": scorer_model_id,
                "scorer_model_name": ALL_MODELS.get(scorer_model_id, {}).get("name", scorer_model_id),
                "scores": parsed,
            }
            scores.append(score_entry)

        session["scores"] = scores
        return {"scores": scores}

    def mutual_score(self, session_id):
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "会话不存在"}

        history = session.get("history", [])
        topic = session.get("topic", "")
        all_model_ids = list(dict.fromkeys(e["model"] for e in history))

        all_scores = []
        calls = []
        call_meta = []
        for scorer_id in all_model_ids:
            identity = ALL_MODELS.get(scorer_id, {}).get("name", scorer_id)
            for idx, entry in enumerate(history):
                if entry["model"] == scorer_id:
                    continue

                speech_text = f"[{entry.get('model_name', entry.get('model'))}"
                side = entry.get("side", "")
                if side:
                    speech_text += f" ({side})"
                content = entry['content']
                if len(content) > 300:
                    content = content[:300] + "…"
                speech_text += f"]: {content}"

                system = MUTUAL_SCORING_PROMPT.format(identity=identity)
                prompt = f"辩论话题：{topic}\n\n请对以下发言进行评分：\n\n{speech_text}"

                calls.append((scorer_id, [{"role": "user", "content": prompt}], system))
                call_meta.append({
                    "speech_index": idx,
                    "speaker_model": entry.get("model"),
                    "speaker_model_name": entry.get("model_name", entry.get("model")),
                    "side": entry.get("side", ""),
                    "scorer_model": scorer_id,
                    "scorer_model_name": ALL_MODELS.get(scorer_id, {}).get("name", scorer_id),
                })

        raw_results = call_models_parallel(calls)

        for i, meta in enumerate(call_meta):
            raw = raw_results[i]
            try:
                parsed = parse_json_response(raw)
                parsed["raw_response"] = raw
            except Exception:
                parsed = {"raw": raw}

            score_entry = {**meta, "scores": parsed}
            all_scores.append(score_entry)

            self._add_event(session_id, {
                "type": "mutual_score",
                **score_entry,
            })

        session["mutual_scores"] = all_scores
        return {"mutual_scores": all_scores}

    def export_text(self, session_id):
        session = self.sessions.get(session_id)
        if not session:
            return None

        lines = []
        lines.append(f"辩论话题: {session['topic']}")
        lines.append(f"辩论模式: {'分组对抗' if session['mode'] == 'group' else '圆桌讨论'}")
        lines.append(f"时间: {session.get('created_at', '')}")
        lines.append("")

        for i, entry in enumerate(session.get("history", []), 1):
            side_str = f" ({entry['side']})" if entry.get("side") else ""
            lines.append(f"--- 第{entry.get('round', '?')}轮 | {entry.get('model_name', entry.get('model'))}{side_str} ---")
            lines.append(entry["content"])
            lines.append("")

        if "summary" in session and session["summary"]:
            s = session["summary"]
            lines.append("=" * 50)
            lines.append(f"总结 (by {s.get('model_name', s.get('model'))}):")
            lines.append(s["content"])
            lines.append("")

        scores = session.get("scores", [])
        mutual_scores = session.get("mutual_scores", [])
        if scores or mutual_scores:
            lines.append("=" * 50)
            lines.append("评分结果")
            lines.append("")

        for sc in scores:
            lines.append(f"  [{sc.get('scorer_model_name', sc.get('scorer_model'))} 评价 {sc.get('model_name', sc.get('model'))}]")
            s = sc.get("scores", {})
            if "论点力度" in s:
                lines.append(f"    论点力度: {s['论点力度']}  逻辑严密: {s['逻辑严密']}  说服力: {s['说服力']}  表达质量: {s['表达质量']}")
            if "总评" in s:
                lines.append(f"    总评: {s['总评']}")
            lines.append("")

        for ms in mutual_scores:
            lines.append(f"  [{ms.get('scorer_model_name', ms.get('scorer_model'))} 评价 {ms.get('speaker_model_name', ms.get('speaker_model'))}]")
            s = ms.get("scores", {})
            if "论点力度" in s:
                lines.append(f"    论点力度: {s['论点力度']}  逻辑严密: {s['逻辑严密']}  说服力: {s['说服力']}  表达质量: {s['表达质量']}")
            if "总评" in s:
                lines.append(f"    总评: {s['总评']}")
            lines.append("")

        return "\n".join(lines)


def _build_system_prompt(role: str, topic: str, side: str = "", round_num: int = 1) -> str:
    base = f"你是一位专业辩手。\n辩论话题：「{topic}」\n你的角色：{role}\n"

    if side == "正方":
        base += (
            "你的立场：正方——支持该辩题/推动现状改变。\n"
            "正方义务：清晰论证辩题成立的理由，提出正面论据，回应反方质疑。\n"
        )
    elif side == "反方":
        base += (
            "你的立场：反方——反对该辩题/维护现状。\n"
            "反方义务：反驳正方论点，提出反面论据，论证辩题不成立。\n"
        )

    base += (
        "\n【辩论规则】\n"
        "1. 立场不可变更：一旦在首轮申论中确立了核心立场，后续发言必须坚持该立场，不能中途倒戈或骑墙。\n"
        "2. 片面性是正常的：辩题将复杂问题极端切割，你只需捍卫己方立场的合理性，不需要追求全面均衡。\n"
        "3. 价值碰撞：好的辩题没有绝对对错，双方代表不同价值维度的碰撞（如效率vs公平、情感vs理智）。\n"
    )

    if round_num == 1:
        base += (
            "\n【本轮要求：一辩申论】\n"
            "这是第一轮，你必须在发言开头明确声明你的核心立场和主要论点框架。\n"
            "格式示例：「我方立场：XXX。核心论点有三：第一……第二……第三……」\n"
        )
    else:
        base += (
            "\n【本轮要求】\n"
            "坚持己方立场，回应对方论点，提出新的支撑论据或反驳。\n"
        )

    base += (
        "\n请用中文回答。控制在200字以内。\n"
        "直接输出辩论发言，不要输出思考过程、分析步骤或任何元评论。\n"
    )
    return base


def _build_messages(
    topic: str, history: list, round_num: int, side: str = ""
) -> list[dict]:
    if not history:
        if side == "正方":
            prompt = (
                f"辩论话题：{topic}\n\n"
                f"你是正方（支持辩题/推动改变），请在首轮申论中清晰界定你的核心立场，"
                f"并陈述你的主要论点框架。"
            )
        elif side == "反方":
            prompt = (
                f"辩论话题：{topic}\n\n"
                f"你是反方（反对辩题/维护现状），请在首轮申论中清晰界定你的核心立场，"
                f"并陈述你的主要论点框架。"
            )
        else:
            prompt = (
                f"辩论话题：{topic}\n\n"
                f"请开始你的第一轮发言，清晰陈述你的立场和主要论点。"
            )
        return [{"role": "user", "content": prompt}]

    # Compress history: combine all previous speeches into a single user message
    # to keep context short and avoid the model seeing fake "assistant" turns
    history_lines = []
    for entry in history:
        label = entry.get("model_name", entry.get("model"))
        side_tag = f"（{entry['side']}）" if entry.get("side") else ""
        # Truncate each speech to 150 chars to keep context manageable
        content = entry['content']
        if len(content) > 150:
            content = content[:150] + "…"
        history_lines.append(f"{label}{side_tag}：{content}")

    history_text = "\n".join(history_lines)

    side_str = f"（{side}）" if side else ""
    if round_num == 1:
        prompt = (
            f"以下是之前的讨论记录：\n{history_text}\n\n"
            f"请针对以上讨论内容{side_str}，发表你的首轮申论，"
            f"明确界定你的核心立场和主要论点框架。"
        )
    else:
        prompt = (
            f"以下是之前的讨论记录：\n{history_text}\n\n"
            f"请针对以上讨论内容{side_str}，发表你在第{round_num}轮的观点。"
            f"坚持己方立场，回应对方论点。"
        )
    return [{"role": "user", "content": prompt}]


def _format_history(history: list) -> str:
    parts = []
    for e in history:
        label = e.get("model_name", e.get("model"))
        side = e.get("side", "")
        side_str = f" ({side})" if side else ""
        content = e['content']
        # Truncate for summary context too
        if len(content) > 300:
            content = content[:300] + "…"
        parts.append(f"[{label}{side_str}]: {content}")
    return "\n\n".join(parts)