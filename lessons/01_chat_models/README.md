# Lesson 01 · Chat models

## What you'll learn

- The three core message types: `SystemMessage`, `HumanMessage`, `AIMessage`
- How to send raw messages vs. how to use a `ChatPromptTemplate`
- How `MessagesPlaceholder` slots conversation history into a prompt
- That a `prompt | model` is your first **LCEL chain** (the pipe operator returns a Runnable)

## Why it matters

A chat model is a function from `list[Message]` to `AIMessage`. Once
you internalise that, prompts and chains are just convenient ways of
building the input list. Templates make prompts reusable across calls;
placeholders make them history-aware.

## Key concepts

- **Messages** — typed wrappers with a `content` field and a role (system/human/ai/tool).
- **`ChatPromptTemplate`** — a list of message templates with `{variables}`. Calling `.invoke({...})` returns the rendered messages.
- **`MessagesPlaceholder("history")`** — a slot you fill with a list of messages at call time. Essential for multi-turn chat.
- **Pipe operator (`|`)** — connects two Runnables. `prompt | llm` is a Runnable that takes the prompt's input and yields the model's output.

## Walk through `example.py`

The script has three small parts:

1. **`raw_messages()`** — proves that a chat model just consumes a list of messages. No template, no chain.
2. **`with_prompt_template()`** — same idea but parameterised. Now you can call the same chain with different `language` and `text` values.
3. **`with_message_history()`** — adds `MessagesPlaceholder("history")` so the model sees prior turns. This is the seed of every chat agent you'll build later.

## Run it

```bash
uv run python -m lessons.01_chat_models.example
```

## Debug it

Put `breakpoint()` after `prompt.invoke({...})` returns to inspect the rendered messages list — it's the cleanest way to see what the model will actually receive.

## Try it yourself

- Add an `AIMessage` to the history in part 3 and ask a follow-up that references it.
- Replace the system message in part 1 with something more elaborate and see how it changes tone.

## Next →

[Lesson 02 · LCEL chains](../02_lcel_chains/README.md)
