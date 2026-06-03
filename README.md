# AI Assistant with Tool Calling

A small AI assistant that decides for itself when to answer from its own
knowledge and when to call an external tool. Built as an AI Engineer technical
case.

- "Quem foi Albert Einstein?" gets answered from the model's knowledge.
- "Quanto é 128 × 46?" calls a calculator tool so the result is exact.
- "What's the weather in Lisbon?" calls a weather tool (an optional extra).

It runs through [OpenRouter](https://openrouter.ai), an OpenAI-compatible API, so
the same code can compare two different models on the same question.

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env      # then put your OpenRouter key in .env
```

Ask one model:

```bash
python assistant.py "Quanto é 128 vezes 46?"
python assistant.py "Quem foi Albert Einstein?"
```

Compare both models on the same question:

```bash
python assistant.py --compare "Quanto é 128 * 46?"
```

Or open the web UI, which asks both models and shows which tool each one used:

```bash
python assistant.py --web      # http://localhost:5000
```

Example output:

```
$ python assistant.py --compare "Quanto é 128 * 46?"

MODEL_A: openai/gpt-4o-mini
    tool: calculator(128 * 46)=5888
128 vezes 46 é 5888.

MODEL_B: google/gemini-2.5-flash
    tool: calculator(128 * 46)=5888
128 vezes 46 é igual a 5888.
```

## How it works

The decision of when to use a tool is left to the LLM through native function
calling:

1. The question goes to the model along with the schemas of two tools
   (`calculator`, `weather`) and a system prompt telling it to use a tool for
   exact math or live weather and otherwise answer directly.
2. If the model wants a tool, the API returns `tool_calls`. The code runs the
   matching Python function, appends the result to the conversation, and calls
   the model again.
3. This repeats until the model replies with no tool calls. That reply is the
   answer.

Because the model decides, it handles "what is 12% of 340?", "add 5 and 7", or
mixed questions without any hand-written routing. The calculator runs a sandboxed
`eval` (no builtins, character whitelist), so only arithmetic gets executed.

### Why native function calling instead of LangChain

The brief lists LangChain as one option ("...ou mesmo lógica própria"). I
implemented the tool-calling loop directly because this loop is what LangChain's
agents do under the hood: send the tool schemas, read back `tool_calls`, run
them, feed the results in, repeat until the model stops. Writing it by hand keeps
the dependencies small and leaves the decision logic, the part being evaluated,
in plain sight. LangChain pays off once you need many tools, retrieval, memory,
or multi-step planning. At this size it adds weight without adding clarity.

### Streaming

Answers stream token by token. A tool call has to arrive complete before it can
run, so `stream_answer()` collects the tool call from the stream, runs it, then
streams the model's final answer. The CLI prints tokens as they arrive. The web
UI sends newline-delimited JSON events (`start`, `tool`, `text`, `end`) and the
page appends them to each model's card.

### Comparing two models

Comparison runs the same `stream_answer()` loop against two model ids from
`.env`. It is a quick way to see how different models handle the same
tool-calling task. The web UI (`--web`) is a single-file Flask app: the HTML
lives inside `assistant.py`, an `/ask` endpoint runs both models, and the page
shows each answer and the tools it used.

## What I'd do differently with more time

- Replace the sandboxed `eval` with a real expression parser (Python's `ast`).
- Add more tools (unit conversion, public APIs) and let the model chain a few in
  one answer.
- Show latency and token cost per model in compare mode.
- Add tests for the calculator edge cases and the tool-call accumulation.
