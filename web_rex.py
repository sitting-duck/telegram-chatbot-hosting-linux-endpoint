# web_rex.py — simple two-column web demo for Rex (RAG vs no-RAG)

from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Objects we expect via ctx from main.py:
#  - stream_ollama: async generator(prompt, sys_prompt=...)
#  - BM25, DOCS, bm25_retrieve
#  - MAX_TOTAL_CHARS, MIN_BM25_SCORE, MAX_CONTEXT_DOCS
#  - categorize, log_interaction
#  - (optional) BM25_TOPK, RERANK_TOPN, rerank_candidates

EXAMPLE_PROMPTS: List[str] = [
    "I’m new to preparedness. What are 10 cheap items I should buy this week to be more resilient at home?",
    "Bushcraft question: How can I build a simple overnight shelter in the woods with just a tarp and some cordage?",
    "If the power grid went down in my area for several days, what are the first things I should do in the first 24 hours?",
]


def json_escape_for_js(arr: List[str]) -> str:
    """Dump a Python list to JSON for safe embedding in JS."""
    import json
    return json.dumps(arr, ensure_ascii=False)


async def _run_ollama(
    stream_ollama,
    user_prompt: str,
    sys_prompt: str | None,
    max_chars: int,
) -> str:
    """Consume the streaming generator into a single string, with a hard char cap."""
    chunks: List[str] = []
    total = 0
    async for chunk in stream_ollama(user_prompt, sys_prompt=sys_prompt):
        if not chunk:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    text = "".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n…(truncated for length)"
    return text or "No response."


def register_web_routes(app: FastAPI, ctx: Dict[str, Any]) -> None:
    stream_ollama = ctx["stream_ollama"]
    BM25 = ctx.get("BM25")
    DOCS = ctx.get("DOCS") or []
    bm25_retrieve = ctx["bm25_retrieve"]
    MAX_TOTAL_CHARS = int(ctx.get("MAX_TOTAL_CHARS", 3500))
    MIN_BM25_SCORE = float(ctx.get("MIN_BM25_SCORE", 2.0))
    MAX_CONTEXT_DOCS = int(ctx.get("MAX_CONTEXT_DOCS", 5))
    categorize = ctx["categorize"]
    log_interaction = ctx["log_interaction"]

    BM25_TOPK = int(ctx.get("BM25_TOPK", 50))
    RERANK_TOPN = int(ctx.get("RERANK_TOPN", 3))
    rerank_candidates = ctx.get("rerank_candidates")  # may be None

    # --- HTML UI ---
    @app.get("/", response_class=HTMLResponse)
    async def rex_page(request: Request) -> HTMLResponse:
        examples_json = json_escape_for_js(EXAMPLE_PROMPTS)

        html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rex – Survival RAG Demo</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0b1020;
      color: #f5f5f5;
      margin: 0;
      padding: 0;
    }}
    header {{
      padding: 16px 24px;
      border-bottom: 1px solid #222b45;
      background: #0f1428;
    }}
    h1 {{
      margin: 0 0 4px 0;
      font-size: 24px;
    }}
    header p {{
      margin: 0;
      font-size: 13px;
      color: #9ea4c7;
    }}
    main {{
      padding: 16px 24px 24px 24px;
    }}
    #prompt-bar {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 16px;
    }}
    #user-input {{
      width: 100%;
      min-height: 70px;
      max-height: 160px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid #303a5b;
      background: #0b1020;
      color: #f5f5f5;
      resize: vertical;
      font-family: inherit;
    }}
    #controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: flex-start;
    }}
    button {{
      border-radius: 999px;
      border: 1px solid #4651a7;
      padding: 6px 12px;
      background: #1b2550;
      color: #f5f5f5;
      font-size: 13px;
      cursor: pointer;
    }}
    button:hover {{
      background: #232f66;
    }}
    button:disabled {{
      opacity: 0.5;
      cursor: default;
    }}
    #columns {{
      display: flex;
      flex-direction: row;
      gap: 16px;
      margin-top: 12px;
    }}
    .column {{
      flex: 1;
      min-width: 0;
      background: #10162f;
      border-radius: 12px;
      padding: 12px 14px;
      border: 1px solid #202a4a;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .column h2 {{
      margin: 0 0 4px 0;
      font-size: 16px;
    }}
    .output {{
      white-space: pre-wrap;
      font-size: 14px;
      line-height: 1.4;
      background: #080c19;
      border-radius: 8px;
      padding: 8px 10px;
      border: 1px solid #1c2440;
      min-height: 80px;
    }}
    .sources {{
      margin-top: 4px;
      padding: 0;
      list-style: none;
      font-size: 13px;
      max-height: 200px;
      overflow-y: auto;
    }}
    .sources li {{
      margin-bottom: 6px;
      border-bottom: 1px solid #202a4a;
      padding-bottom: 4px;
    }}
    .sources code {{
      font-size: 12px;
    }}
    .pill {{
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid #37406c;
      color: #9ea4c7;
      display: inline-block;
      margin-left: 6px;
    }}
    @media (max-width: 900px) {{
      #columns {{
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Rex – Survival RAG Demo</h1>
    <p>
      Left: <strong>Rex + Survival Data (RAG)</strong> • Right: <strong>Model only (no RAG)</strong>.
      One question, two answers.
    </p>
  </header>
  <main>
    <section id="prompt-bar">
      <label for="user-input" style="font-size:13px;color:#c0c5eb;">Ask Rex a question:</label>
      <textarea id="user-input" placeholder="Example: What should I keep in a blackout emergency kit for a toddler?" autofocus></textarea>
      <div id="controls">
        <button id="ask-btn" onclick="sendQuery()">Ask Rex</button>
        <span style="font-size:12px;color:#9ea4c7;">Or try an example:</span>
        <button type="button" onclick="useExample(0)">Example 1</button>
        <button type="button" onclick="useExample(1)">Example 2</button>
        <button type="button" onclick="useExample(2)">Example 3</button>
        <span id="status" style="font-size:12px;color:#9ea4c7;margin-left:auto;"></span>
      </div>
    </section>

    <section id="columns">
      <div class="column">
        <h2>Rex + Survival Data (RAG) <span class="pill">BM25 + reranker</span></h2>
        <div id="rag-output" class="output">Ask a question to see Rex’s answer with Survival-Data context.</div>
        <h3 style="margin:4px 0 2px 0;font-size:13px;">Sources</h3>
        <ul id="sources-list" class="sources"></ul>
      </div>

      <div class="column">
        <h2>Model only (no RAG) <span class="pill">Baseline</span></h2>
        <div id="norag-output" class="output">Ask a question to see the no-RAG answer for comparison.</div>
      </div>
    </section>
  </main>

  <script>
    const EXAMPLES = {examples_json};

    function useExample(idx) {{
      const ta = document.getElementById('user-input');
      ta.value = EXAMPLES[idx] || '';
      ta.focus();
      ta.selectionStart = ta.selectionEnd = ta.value.length;
      sendQuery();
    }}

    async function sendQuery() {{
      const inputEl = document.getElementById('user-input');
      const askBtn = document.getElementById('ask-btn');
      const statusEl = document.getElementById('status');
      const ragEl = document.getElementById('rag-output');
      const noragEl = document.getElementById('norag-output');
      const sourcesEl = document.getElementById('sources-list');

      const query = inputEl.value.trim();
      if (!query) {{
        alert('Please enter a question first.');
        return;
      }}

      askBtn.disabled = true;
      statusEl.textContent = 'Thinking…';
      ragEl.textContent = 'Rex is reading the Survival-Data corpus…';
      noragEl.textContent = 'Model is answering without external documents…';
      sourcesEl.innerHTML = '';

      try {{
        const resp = await fetch('/api/rex', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ query }})
        }});
        if (!resp.ok) {{
          const text = await resp.text();
          throw new Error('HTTP ' + resp.status + ': ' + text);
        }}
        const data = await resp.json();
        ragEl.textContent = data.rag_answer || '(no RAG answer)';
        noragEl.textContent = data.norag_answer || '(no no-RAG answer)';

        const sources = data.sources || [];
        if (sources.length === 0) {{
          sourcesEl.innerHTML = '<li>No documents were confident enough to include.</li>';
        }} else {{
          sourcesEl.innerHTML = '';
          sources.forEach((src, idx) => {{
            const li = document.createElement('li');
            const title = src.title || '(no title)';
            const score = typeof src.score === 'number'
              ? src.score.toFixed(3)
              : String(src.score || '');
            const sourcePath = src.source || '';
            const snippet = src.snippet || '';
            li.innerHTML = '<strong>#' + (idx+1) + '</strong> ' + title +
              (sourcePath ? ' <br><code>' + sourcePath + '</code>' : '') +
              (score ? ' &nbsp; <span class="pill">score ' + score + '</span>' : '') +
              (snippet ? '<br><span>' + snippet + '</span>' : '');
            sourcesEl.appendChild(li);
          }});
        }}
        statusEl.textContent = 'Done.';
      }} catch (err) {{
        console.error(err);
        statusEl.textContent = 'Error.';
        ragEl.textContent = 'Error: ' + err;
        noragEl.textContent = '';
        sourcesEl.innerHTML = '';
      }} finally {{
        askBtn.disabled = false;
      }}
    }}
  </script>
</body>
</html>
""".format(examples_json=examples_json)

        return HTMLResponse(html)

    # --- API: /api/rex ---
    @app.post("/api/rex")
    async def api_rex(payload: Dict[str, Any]) -> JSONResponse:
        query = (payload.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "missing 'query'"}, status_code=400)

        if BM25 is None or not DOCS:
            # RAG index missing – just call model twice
            concise_sys_prompt = (
                "You are a helpful, concise assistant replying for a web demo. "
                "Keep answers under ~200 words unless the user asks for more detail."
            )
            rag_answer = await _run_ollama(stream_ollama, query, concise_sys_prompt, MAX_TOTAL_CHARS)
            norag_answer = rag_answer  # identical when no corpus
            return JSONResponse({"rag_answer": rag_answer, "norag_answer": norag_answer, "sources": []})

        # 1) BM25 retrieve
        candidates = bm25_retrieve(query, BM25, DOCS, top_k=BM25_TOPK)
        best_score = candidates[0]["score"] if candidates else 0.0

        chosen_docs: List[Dict[str, Any]] = []
        if candidates and best_score >= MIN_BM25_SCORE:
            if rerank_candidates is not None:
                # 2) CrossEncoder rerank
                chosen_docs = rerank_candidates(query, candidates, top_k=MAX_CONTEXT_DOCS)
            else:
                chosen_docs = candidates[:MAX_CONTEXT_DOCS]

        # Build context block
        context_pieces: List[str] = []
        for d in chosen_docs:
            title = d.get("title") or ""
            prefix = f"[{title}] " if title else ""
            context_pieces.append(prefix + (d.get("text") or ""))
        context_block = "\n\n".join(context_pieces)

        survival_sys_prompt = (
            "You are Rex, an experienced survival and preparedness instructor talking to a beginner. "
            "Be concise, practical, and safety-focused. Use short paragraphs or bullets for steps. "
            "Prefer concrete, real-world advice about bushcraft, homesteading, emergency preparedness, "
            "medical and safety basics, and gear usage. If something is highly uncertain or depends on "
            "local regulations, say so explicitly."
        )

        if context_block:
            rag_user_prompt = (
                f"User question:\n{query}\n\n"
                "Here are relevant excerpts from a survival knowledge base. "
                "Use them when they clearly help answer the question:\n\n"
                f"{context_block}\n\n"
                "Now answer the question above as Rex, using the excerpts plus your own survival expertise. "
                "Do NOT mention 'documents' or 'context'; just answer normally."
            )
        else:
            rag_user_prompt = (
                f"The user asked:\n{query}\n\n"
                "Answer as Rex the survival instructor using your general survival knowledge. "
                "If something is uncertain or risky, explain the tradeoffs and safety precautions."
            )

        # RAG answer
        rag_answer = await _run_ollama(
            stream_ollama,
            rag_user_prompt,
            survival_sys_prompt,
            MAX_TOTAL_CHARS,
        )

        # No-RAG baseline answer
        concise_sys_prompt = (
            "You are a helpful, concise assistant replying for a web demo. "
            "Keep answers under ~200 words unless the user asks for more detail."
        )
        norag_answer = await _run_ollama(
            stream_ollama,
            query,
            concise_sys_prompt,
            MAX_TOTAL_CHARS,
        )

        # Build sources for the panel
        sources_payload: List[Dict[str, Any]] = []
        for d in chosen_docs:
            text = d.get("text") or ""
            snippet = text[:300].replace("\n", " ")
            if len(text) > 300:
                snippet += " …"
            sources_payload.append({
                "title": d.get("title") or "",
                "source": d.get("source_path") or d.get("id") or "",
                "score": float(d.get("score", 0.0)),
                "snippet": snippet,
            })

        # Logging
        try:
            log_interaction(
                user_id=0,  # web demo: no Telegram user id
                message=query,
                reply_len=len(rag_answer),
                response_time_ms=0,  # could be measured if we wanted
                category=categorize(query),
                error=False,
                meta={"mode": "web_rag_vs_norag"},
            )
        except Exception:
            pass

        return JSONResponse(
            {
                "rag_answer": rag_answer,
                "norag_answer": norag_answer,
                "sources": sources_payload,
            }
        )

