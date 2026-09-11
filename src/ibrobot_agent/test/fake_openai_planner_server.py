"""One-purpose local model server for the SO-101 hardware E2E trial."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PLANNED_SKILL = "nod_yes"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if request.get("model") != "deterministic-so101-e2e" or not request.get("messages"):
            self.send_error(400)
            return
        prompt_text = request["messages"][-1].get("content", "")
        try:
            request_text = json.loads(prompt_text).get("request", {}).get("text", "")
        except (TypeError, json.JSONDecodeError):
            request_text = ""
        folded = str(request_text).casefold()
        if any(token in folded for token in ("不要", "别", "不想", "do not", "don't")):
            outcome = {
                "kind": "conversation",
                "user_message": "我不会执行被明确否定的动作。",
            }
        elif any(token in folded for token in ("当前状态", "状态怎么样", "status")):
            outcome = {
                "kind": "read_only",
                "user_message": "我来查看当前机器人的状态。",
                "query_kind": "status",
            }
        elif any(token in folded for token in ("有哪些能力", "有哪些技能", "能力列表", "list skills")):
            outcome = {
                "kind": "read_only",
                "user_message": "我来查看当前机器人的能力。",
                "query_kind": "list_skills",
            }
        else:
            aliases = {
                "nod_yes": ("点头", "同意", "nod", "yes"),
                "wave_hello": ("挥手", "打招呼", "wave", "hello"),
                "shake_no": ("摇头", "不同意", "shake", "no"),
                "recover_safe_pose": ("回安全位", "安全位", "safe pose"),
                "open_gripper_skill": ("打开夹爪", "开爪", "open gripper"),
                "close_gripper_skill": ("关闭夹爪", "关爪", "close gripper"),
            }
            selected = next(
                (skill for skill, skill_aliases in aliases.items() if any(alias in folded for alias in skill_aliases)),
                None,
            )
            if selected is None:
                outcome = {
                    "kind": "rejected",
                    "user_message": "我无法把这句话安全映射到已启用的 SO-101 技能。",
                    "reason_code": "UNSUPPORTED_TASK",
                }
            else:
                outcome = {
                    "kind": "workflow",
                    "user_message": f"准备执行：{selected}。",
                    "summary": selected,
                    "steps": [{"schema_version": 1, "skill_name": selected}],
                }
        content = json.dumps(outcome, ensure_ascii=False, separators=(",", ":"))
        body = json.dumps(
            {
                "choices": [{"message": {"content": content, "tool_calls": []}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            ensure_ascii=False,
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args) -> None:
        return


def main() -> None:
    global PLANNED_SKILL

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--skill-name", choices=("nod_yes", "wave_hello"), default="nod_yes")
    args = parser.parse_args()
    PLANNED_SKILL = args.skill_name
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
