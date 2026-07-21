from flask import Flask, render_template, request

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from langchain_core.utils.uuid import uuid7

import os
import json
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')

SYSTEM_PROMPT = """You are a personal assistant with persistent memory.

  Your persistent memory file lives at /memories/profile.md and survives
  across all conversations.

  When to update memory:
  - User shares name, role, or background
  - User mentions ongoing projects or goals
  - User states a preference (language, tools, response format)
  - User corrects you or gives explicit feedback

  How to update:
  - First conversation: write_file to create memories/profile.md
  - Later conversations: edit_file to update it

  Keep the file concise — bullet points, not prose.
  Never store credentials.

  Always answer in Spanish.
"""

store = InMemoryStore()

store_backend = StoreBackend(
  store=store,
  namespace=lambda rt: (f"user:engineering", "memories"),
)


agent = create_deep_agent(
    model="google_genai:gemini-2.5-flash",
    system_prompt=SYSTEM_PROMPT,
    memory=["/memories/profile.md"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": FilesystemBackend(root_dir=f"{os.path.join(app.root_path, 'memories')}", virtual_mode=True)},
    ),
    store=InMemoryStore(),
)


@app.get("/")
def index_get():
    try:
        with open(f"{os.path.join(app.root_path, 'messages/messages.json')}", "rt") as f:
            messages = json.load(f)
    except FileNotFoundError:
        messages = []

    return render_template("index.html", messages=messages, thread_id=str(uuid7()))


@app.post("/")
def index_post():
    if request.form.get("reset_button"):
        os.remove(f"{os.path.join(app.root_path, 'messages/messages.json')}")
        os.remove(f"{os.path.join(app.root_path, 'memories/profile.md')}")
        return render_template("index.html", messages=[], show_tool_messages=show_tool_messages, thread_id=thread_id)

    user_input = request.form.get("user_input")
    thread_id = request.form.get("thread_id")
    show_tool_messages = request.form.get("show_tool_messages", False)

    try:
        with open(f"{os.path.join(app.root_path, 'messages/messages.json')}", "rt") as f:
            messages = json.load(f)
    except FileNotFoundError:
        messages = []

    if user_input == "":
        return render_template("index.html", messages=messages, show_tool_messages=show_tool_messages, thread_id=thread_id)

    res = agent.invoke({ "messages": [HumanMessage(content=user_input)] },
                       config={"configurable": {"thread_id": thread_id}},
                       version="v3")

    for message in res["messages"]:
        if message.type == "human":
            if message.content != "":
                messages.append({"type": message.type,"content": message.content})
        elif message.type == "ai":
            if type(message.content) == list:
                content = "<br>".join([c["text"] for c in message.content if c["text"] != ""])
            else:
                content = message.content

            if message.content != "":
                messages.append({"type": message.type, "content": content, "total_tokens": message.usage_metadata.get("total_tokens")})
        else:
            messages.append({"type": message.type, "content": message.content})
            print(message)

    with open(f"{os.path.join(app.root_path, 'messages/messages.json')}", "wt") as f:
        f.write(json.dumps(messages, indent=2))

    return render_template("index.html", messages=messages, show_tool_messages=show_tool_messages, thread_id=thread_id)


if __name__ == "__main__":
    app.run(debug=True)