"""
A small AI assistant that decides on its own when to answer from its own
knowledge and when to call an external tool (calculator / weather).

The decision is made by the LLM via native function calling. We run on
OpenRouter (OpenAI-compatible API) so the same question can be sent through two
different models and compared.

    python assistant.py "Quem foi Albert Einstein?"   # one model
    python assistant.py --compare "Quanto é 128 * 46?" # both models
    python assistant.py --web                          # browser UI
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL_A = os.getenv("MODEL_A", "openai/gpt-4o-mini")
MODEL_B = os.getenv("MODEL_B", "google/gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer general questions from your own "
    "knowledge. When a question needs exact math or live weather, call the "
    "matching tool instead of guessing. Keep answers short."
)


# ---------------------------------------------------------------------------
# Tools the model can call: calculator and weather
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '128 * 46'."""
    if set(expression) - set("0123456789+-*/(). %"):
        return "error: expression contains characters that are not allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # arithmetic only
    except Exception as exc:
        return f"error: {exc}"


def weather(city: str) -> str:
    """Current temperature for a city, via the free open-meteo API (no key)."""
    try:
        matches = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10,
        ).json().get("results")
        if not matches:
            return f"error: city '{city}' not found"

        place = matches[0]
        current = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": place["latitude"], "longitude": place["longitude"],
                    "current": "temperature_2m"}, timeout=10,
        ).json()["current"]
        return f"{current['temperature_2m']}°C in {place['name']}, {place.get('country', '')}"
    except Exception as exc:
        return f"error: {exc}"


TOOLBOX = {"calculator": calculator, "weather": weather}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Compute the exact result of an arithmetic expression. "
                           "Use this for ANY math question instead of answering yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '128 * 46'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the current temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "e.g. 'Lisbon'"},
                },
                "required": ["city"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Core: stream an answer, running tools when the model asks for them
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A tool call the model is requesting, assembled from streamed fragments."""
    id: str = ""
    name: str = ""
    arguments: str = ""

    def absorb(self, fragment):
        self.id += fragment.id or ""
        if fragment.function:
            self.name += fragment.function.name or ""
            self.arguments += fragment.function.arguments or ""


def run_tool(call: ToolCall):
    """Run one tool call. Returns (result, human_readable_summary)."""
    args = json.loads(call.arguments or "{}")
    result = TOOLBOX[call.name](**args)
    summary = f"{call.name}({', '.join(map(str, args.values()))})={result}"
    return result, summary


def assistant_turn(text, calls):
    """The assistant message recording the tool calls it asked for."""
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ],
    }


def tool_result(call, result):
    """The message that feeds a tool's result back to the model."""
    return {"role": "tool", "tool_call_id": call.id, "content": result}


def stream_completion(model, messages):
    """Stream one model turn. Yields ('text', chunk) as the answer arrives,
    and returns (full_text, tool_calls). tool_calls may be empty."""
    stream = client.chat.completions.create(
        model=model, messages=messages, tools=TOOL_SCHEMAS, stream=True,
    )
    text = ""
    calls: dict[int, ToolCall] = {}
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            text += delta.content
            yield "text", delta.content
        for fragment in delta.tool_calls or []:
            calls.setdefault(fragment.index, ToolCall()).absorb(fragment)
    return text, list(calls.values())


def stream_answer(model, question):
    """Yield events for one model:
        ('tool', summary)  when a tool runs
        ('text', chunk)    as the final answer streams in
    Loops until the model returns an answer with no further tool calls."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    while True:
        # `yield from` forwards the 'text' events AND captures the return value.
        text, calls = yield from stream_completion(model, messages)
        if not calls:
            return

        messages.append(assistant_turn(text, calls))
        for call in calls:
            result, summary = run_tool(call)
            yield "tool", summary
            messages.append(tool_result(call, result))


def answer(model, question):
    """Non-streaming helper: returns (text, tools_used)."""
    text, tools_used = "", []
    for kind, payload in stream_answer(model, question):
        if kind == "text":
            text += payload
        else:
            tools_used.append(payload)
    return text, tools_used


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_answer(label, model, question):
    print(f"{label}: {model}")
    used_tool = False
    for kind, payload in stream_answer(model, question):
        if kind == "tool":
            print(f"    tool: {payload}")
            used_tool = True
        else:
            sys.stdout.write(payload)
            sys.stdout.flush()
    if not used_tool:
        print("    (answered from knowledge)", end="")
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Tool-calling AI assistant (OpenRouter)")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--compare", action="store_true", help="Ask both MODEL_A and MODEL_B")
    parser.add_argument("--web", action="store_true", help="Launch the web UI at :5000")
    args = parser.parse_args()

    if args.web:
        run_web()
        return

    question = args.question or input("Question: ")
    print()
    print_answer("MODEL_A", MODEL_A, question)
    if args.compare:
        print_answer("MODEL_B", MODEL_B, question)


# ---------------------------------------------------------------------------
# Web UI: single-file Flask app that streams events as newline-delimited JSON
# ---------------------------------------------------------------------------

INDEX_HTML = """
<!doctype html><html lang="en"><meta charset="utf-8">
<title>AI Assistant</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#1a1a1a}
  h1{font-size:1.3rem}
  form{display:flex;gap:8px;margin:20px 0}
  input{flex:1;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:1rem}
  button{padding:10px 18px;border:0;border-radius:8px;background:#3b5bdb;color:#fff;font-size:1rem;cursor:pointer}
  button:disabled{opacity:.5}
  .card{border:1px solid #e3e3e3;border-radius:10px;padding:14px 16px;margin:12px 0}
  .model{font-size:.8rem;color:#666;margin-bottom:6px}
  .tag{font-size:.8rem;border-radius:6px;padding:4px 8px;display:inline-block;margin-bottom:8px}
  .tool{color:#0a7d33;background:#eefaf0}
  .knowledge{color:#666;background:#f0f0f0}
  .answer{white-space:pre-wrap}
  .hint{font-size:.85rem;color:#666}
</style>
<h1>AI Assistant</h1>
<form id="form">
  <input id="question" placeholder="Quanto é 128 * 46?  /  Quem foi Albert Einstein?" autofocus>
  <button id="submit">Ask both models</button>
</form>
<p class="hint">It answers general questions itself and calls a tool for exact math or live weather. Try one of each.</p>
<div id="output"></div>
<script>
const form = document.getElementById('form');
const questionBox = document.getElementById('question');
const submitBtn = document.getElementById('submit');
const output = document.getElementById('output');
const escapeHtml = s => s.replace(/</g, '&lt;');
const cards = {};  // label -> { answer, tag } DOM nodes

function addCard(event) {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `<div class="model">${event.label}: ${event.model}</div>`
    + `<div class="tag" style="display:none"></div><div class="answer"></div>`;
  output.appendChild(card);
  cards[event.label] = {
    tag: card.querySelector('.tag'),
    answer: card.querySelector('.answer'),
    usedTool: false,
  };
}

function showTool(event) {
  const card = cards[event.label];
  card.usedTool = true;
  card.tag.className = 'tag tool';
  card.tag.style.display = 'inline-block';
  card.tag.textContent = 'tool: ' + event.info;
}

function appendText(event) {
  cards[event.label].answer.innerHTML += escapeHtml(event.delta);
}

function finish(event) {
  const card = cards[event.label];
  if (card.usedTool) return;
  card.tag.className = 'tag knowledge';
  card.tag.style.display = 'inline-block';
  card.tag.textContent = 'answered from knowledge';
}

const handlers = { start: addCard, tool: showTool, text: appendText, end: finish };

form.onsubmit = async (e) => {
  e.preventDefault();
  if (!questionBox.value.trim()) return;
  submitBtn.disabled = true;
  output.innerHTML = '';
  for (const key in cards) delete cards[key];

  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: questionBox.value }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        handlers[event.type](event);
      }
    }
  } catch (err) {
    output.innerHTML += '<p style="color:#c00">Error: ' + err + '</p>';
  }
  submitBtn.disabled = false;
};
</script></html>
"""


def run_web():
    from flask import Flask, request, Response

    app = Flask(__name__)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.post("/ask")
    def ask():
        question = (request.json or {}).get("question", "")

        def events():
            for label, model in [("MODEL_A", MODEL_A), ("MODEL_B", MODEL_B)]:
                yield emit("start", label, model=model)
                for kind, payload in stream_answer(model, question):
                    field_name = "info" if kind == "tool" else "delta"
                    yield emit(kind, label, **{field_name: payload})
                yield emit("end", label)

        return Response(events(), mimetype="application/x-ndjson")

    print("Web UI → http://localhost:5000")
    app.run(port=5000, threaded=True)


def emit(event_type, label, **fields):
    """One newline-delimited JSON event for the streaming response."""
    return json.dumps({"type": event_type, "label": label, **fields}) + "\n"


if __name__ == "__main__":
    main()
