import json
import uuid
import threading
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from debate import DebateManager
from api_client import set_custom_api

app = Flask(__name__, template_folder="templates", static_folder="static")
manager = DebateManager()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def get_models():
    from config import ALL_MODELS

    models = []
    for mid, info in ALL_MODELS.items():
        models.append(
            {
                "id": mid,
                "name": info["name"],
                "emoji": info["emoji"],
                "color": info["color"],
            }
        )
    return jsonify(models)


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json
    api_key = data.get("api_key", "").strip()
    base_url = data.get("base_url", "").strip()
    set_custom_api(api_key or None, base_url or None)
    return jsonify({"status": "ok"})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    from api_client import _custom_api_key, _custom_base_url
    return jsonify({
        "has_custom_key": _custom_api_key is not None,
        "has_custom_url": _custom_base_url is not None,
    })


@app.route("/api/debate/start", methods=["POST"])
def start_debate():
    data = request.json
    mode = data.get("mode", "roundtable")
    topic = data.get("topic", "").strip()
    rounds = data.get("rounds", 3)
    models = data.get("models")
    group_a = data.get("group_a")
    group_b = data.get("group_b")

    if not topic:
        return jsonify({"error": "话题不能为空"}), 400

    session_id = str(uuid.uuid4())[:8]

    if mode == "roundtable":
        if not models:
            from config import DEFAULT_DEBATE_MODELS
            models = DEFAULT_DEBATE_MODELS
        t = threading.Thread(
            target=manager.run_roundtable,
            args=(session_id, topic, models, rounds),
            daemon=True,
        )
    else:
        if not group_a:
            from config import DEFAULT_GROUP_A
            group_a = DEFAULT_GROUP_A
        if not group_b:
            from config import DEFAULT_GROUP_B
            group_b = DEFAULT_GROUP_B
        t = threading.Thread(
            target=manager.run_group,
            args=(session_id, topic, group_a, group_b, rounds),
            daemon=True,
        )

    t.start()
    return jsonify({"session_id": session_id})


@app.route("/api/debate/stream/<session_id>")
def debate_stream(session_id):
    def generate():
        while True:
            event = manager.get_next_event(session_id)
            if event is None:
                import time
                time.sleep(0.5)
                if manager.is_finished(session_id):
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "done":
                break

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/debate/summary", methods=["POST"])
def generate_summary():
    data = request.json
    session_id = data.get("session_id")
    model_id = data.get("model_id")
    summary_type = data.get("summary_type", "auto")

    if not session_id or not model_id:
        return jsonify({"error": "参数缺失"}), 400

    history = manager.get_history(session_id)
    if not history:
        return jsonify({"error": "辩论记录不存在"}), 404

    result = manager.generate_summary(session_id, model_id, summary_type)
    return jsonify(result)


@app.route("/api/debate/score", methods=["POST"])
def score_speeches():
    data = request.json
    session_id = data.get("session_id")
    scorer_model_id = data.get("model_id")
    speech_indices = data.get("speech_indices")

    if not session_id or not scorer_model_id:
        return jsonify({"error": "参数缺失"}), 400

    history = manager.get_history(session_id)
    if not history:
        return jsonify({"error": "辩论记录不存在"}), 404

    result = manager.score_speeches(session_id, scorer_model_id, speech_indices)
    return jsonify(result)


@app.route("/api/debate/mutual-score", methods=["POST"])
def mutual_score():
    data = request.json
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "参数缺失"}), 400

    history = manager.get_history(session_id)
    if not history:
        return jsonify({"error": "辩论记录不存在"}), 404

    def _run():
        manager.mutual_score(session_id)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started", "session_id": session_id})


@app.route("/api/debate/history/<session_id>")
def get_history(session_id):
    history = manager.get_history(session_id)
    if history is None:
        return jsonify({"error": "会话不存在"}), 404
    return jsonify(history)


@app.route("/api/debate/export/<session_id>")
def export_debate(session_id):
    result = manager.export_text(session_id)
    if result is None:
        return jsonify({"error": "会话不存在"}), 404
    return Response(result, mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(debug=True, port=5000)