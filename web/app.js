const $ = (selector) => document.querySelector(selector);

const SUGGESTIONS = [
  "What should I do first?",
  "How do I set up my dev environment?",
  "How does code review work here?",
  "How many leave days do I get?",
];

let busy = false;

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (typeof body.detail === "string") throw new Error(body.detail);
    if (Array.isArray(body.detail)) throw new Error(body.detail.map((d) => d.msg).join("; "));
    throw new Error("Something went wrong. Please try again.");
  }
  return body;
}

function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(text));
  const div = document.createElement("div"); // CDN unavailable: show plain text safely
  div.textContent = text;
  return div.innerHTML.replace(/\n/g, "<br>");
}

function addMessage(kind, text, sources = []) {
  const bubble = document.createElement("div");
  bubble.className = `message ${kind}`;
  if (kind === "assistant") bubble.innerHTML = renderMarkdown(text);
  else bubble.textContent = text;
  if (sources.length) {
    const row = document.createElement("div");
    row.className = "sources";
    row.append("Sources:");
    for (const source of sources) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = source.doc_title;
      chip.title = source.section;
      row.append(chip);
    }
    bubble.append(row);
  }
  $("#messages").append(bubble);
  bubble.scrollIntoView({ block: "end", behavior: "smooth" });
  return bubble;
}

function showSuggestions() {
  $("#suggestions").replaceChildren(
    ...SUGGESTIONS.map((text) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = text;
      button.addEventListener("click", () => send(text));
      return button;
    }),
  );
}

async function refreshChecklist() {
  const status = await api("/api/checklist");
  $("#day-label").textContent =
    status.day >= 1 ? `Day ${status.day} of onboarding` : `You start in ${1 - status.day} day(s)`;
  const percent = status.total ? Math.round((100 * status.completed) / status.total) : 0;
  $("#progress-bar").style.width = `${percent}%`;
  $("#progress-label").textContent = `${status.completed} of ${status.total} tasks done`;

  const byWeek = new Map();
  for (const task of status.tasks) {
    if (!byWeek.has(task.week)) byWeek.set(task.week, []);
    byWeek.get(task.week).push(task);
  }
  const container = $("#checklist");
  container.replaceChildren();
  for (const [week, tasks] of byWeek) {
    const heading = document.createElement("h3");
    heading.textContent = `Week ${week}`;
    const list = document.createElement("ul");
    for (const task of tasks) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = task.done ? "task done" : "task";
      button.textContent = `${task.done ? "✓" : "○"} ${task.title}`;
      button.title = task.done ? "Done: click to ask about it again" : "Ask the buddy to help with this";
      button.addEventListener("click", () =>
        send(task.done ? `Remind me about: ${task.title}` : `Help me with: ${task.title}`),
      );
      item.append(button);
      list.append(item);
    }
    container.append(heading, list);
  }
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  busy = true;
  $("#send").disabled = true;
  $("#chat-input").value = "";
  $("#suggestions").replaceChildren();
  addMessage("user", text);
  const typing = addMessage("typing", "Thinking…");
  try {
    const { reply, sources } = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text }) });
    typing.remove();
    addMessage("assistant", reply, sources);
    await refreshChecklist(); // the agent may have marked a task done
  } catch (error) {
    typing.remove();
    addMessage("error", error.message);
  } finally {
    busy = false;
    $("#send").disabled = false;
    $("#chat-input").focus();
  }
}

async function showApp(employee) {
  $("#setup").hidden = true;
  $("#app").hidden = false;
  $("#greeting").textContent = `Hi ${employee.name} 👋`;
  await refreshChecklist();
  const { messages } = await api("/api/history");
  for (const message of messages) addMessage(message.role, message.content);
  if (!messages.length) {
    addMessage(
      "assistant",
      `Welcome to Nimbus Labs, **${employee.name}**! I'm your Onboarding Buddy. Ask me about your setup, ` +
        "our engineering workflow or company policies, or click a task on the left to get started.",
    );
    showSuggestions();
  }
}

$("#setup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  try {
    const { employee } = await api("/api/employee", { method: "POST", body: JSON.stringify(data) });
    await showApp(employee);
  } catch (error) {
    $("#setup-error").textContent = error.message;
    $("#setup-error").hidden = false;
  }
});

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  send($("#chat-input").value);
});

$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    send(event.target.value);
  }
});

(async function init() {
  const { employee } = await api("/api/employee");
  if (employee) {
    await showApp(employee);
  } else {
    $("#setup").hidden = false;
    $("input[name=start_date]").value = new Date().toLocaleDateString("en-CA"); // local YYYY-MM-DD
  }
})();
