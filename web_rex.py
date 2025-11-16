# web_rex.py — Web UI + JSON API for Rex (2-column compare view + example prompts)

from time import monotonic

from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse


REX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rex Survival Assistant</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0b1020;
      color: #f5f5f5;
      margin: 0;
      padding: 0;
      display: flex;
      min-height: 100vh;
      justify-content: center;
      align-items: center;
    }
    .card {
      background: #161b2b;
      border-radius: 16px;
      padding: 20px;
      width: min(1000px, 96vw);
      box-shadow: 0 18px 45px rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.06);
    }
    h1 {
      margin: 0 0 4px;
      font-size: 1.4rem;
    }
    .subtitle {
      font-size: 0.9rem;
      color: #b3b9d4;
      margin-bottom: 12px;
    }
    .badge {
      display: inline-block;
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 999px;
      background: #1c253f;
      color: #e1e6ff;
      margin-left: 4px;
    }
    .question-label {
      font-size: 0.8rem;
      color: #9aa3c4;
      margin-bottom: 4px;
    }
    textarea {
      width: 100%;
      min-height: 70px;
      max-height: 180px;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid #3a4060;
      background: #070a12;
      color: #f5f5f5;
      resize: vertical;
      font-family: inherit;
      font-size: 0.95rem;
      box-sizing: border-box;
    }
    textarea:focus {
      outline: none;
      border-color: #4f9cff;
      box-shadow: 0 0 0 1px #4f9cff33;
    }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 6px;
      margin-bottom: 4px;
      font-size: 0.8rem;
      color: #9aa3c4;
    }
    .examples-label {
      opacity: 0.9;
    }
    .example-btn {
      border-radius: 999px;
      border: 1px solid #3a4060;
      background: #111626;
      color: #e1e6ff;
      padding: 4px 10px;
      font-size: 0.78rem;
      cursor: pointer;
      white-space: nowrap;
    }
    .example-btn:hover {
      background: #1b2238;
    }
    .top-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 8px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    button {
      border-radius: 999px;
      border: none;
      padding: 8px 16px;
      font-size: 0.9rem;
      cursor: pointer;
      background: #4f9cff;
      color: #050814;
      font-weight: 600;
    }
    button:disabled {
      opacity: 0.6;
      cursor: default;
    }
    .top-status {
      font-size: 0.8rem;
      color: #9aa3c4;
      min-height: 1.1em;
    }
    .columns {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 6px;
    }
    .col {
      flex: 1 1 300px;
      background: #111626;
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid #252b45;
      box-sizing: border-box;
    }
    .col-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 4px;
    }
    .col-title {
      font-size: 0.9rem;
      font-weight: 600;
    }
    .col-pill {
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 999px;
      background: #1f2a4b;
      color: #d0d7ff;
      white-space: nowrap;
    }
    .status {
      font-size: 0.8rem;
      color: #9aa3c4;
      min-height: 1.1em;
      margin-bottom: 4px;
    }
    .answer {
      margin-top: 4px;
      padding: 8px 9px;
      border-radius: 8px;
      background: #070a12;
      white-space: pre-wrap;
      font-size: 0.92rem;
      min-height: 40px;
    }
    .meta {
      font-size: 0.75rem;
      color: #8088aa;
      margin-top: 4px;
    }
    @media (max-width: 720px) {
      .card {
        padding: 16px;
      }
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Rex Survival Assistant <span class="badge">compare</span></h1>
    <div class="subtitle">
      Enter a single question, then compare Rex with Survival-Data (RAG) vs the base model with no RAG.
    </div>

    <div class="question-label">Question for both models</div>
    <textarea id="question" placeholder="Example: What kind of knife should I carry for hiking in the bush?"></textarea>

    <div class="examples">
      <span class="examples-label">Try an example:</span>
      <button class="example-btn" data-example="knife">
        Hiking knife choice
      </button>
      <button class="example-btn" data-example="water">
        Emergency water storage
      </button>
      <button class="example-btn" data-example="wound">
        Treating a deep cut
      </button>
    </div>

    <div class="top-controls">
      <button id="compare-btn">Compare answers (RAG vs no RAG)</button>
      <span class="top-status" id="top-status"></span>
    </div>

    <div class="columns">
      <div class="col">
        <div class="col-header">
          <div class="col-title">Rex with Survival-Data</div>
          <div class="col-pill">RAG mode</div>
        </div>
        <div class="status" id="status-rag"></div>
        <div class="answer" id="answer-rag"></div>
        <div class="meta" id="meta-rag"></div>
      </div>

      <div class="col">
        <div class="col-header">
          <div class="col-title">Base model (no library)</div>
          <div class="col-pill">no-RAG</div>
        </div>
        <div class="status" id="status-norag"></div>
        <div class="answer" id="answer-norag"></div>
        <div class="meta" id="meta-norag"></div>
      </div>
    </div>
  </div>

  <script>
    const questionEl   = document.getElementById('question');
    const compareBtn   = document.getElementById('compare-btn');
    const topStatusEl  = document.getElementById('top-status');

    const statusRagEl    = document.getElementById('status-rag');
    const statusNoragEl  = document.getElementById('status-norag');
    const answerRagEl    = document.getElementById('answer-rag');
    const answerNoragEl  = document.getElementById('answer-norag');
    const metaRagEl      = document.getElementById('meta-rag');
    const metaNoragEl    = document.getElementById('meta-norag');

    const exampleButtons = document.querySelectorAll('.example-btn');

    // Example prompt text — you can tweak these to match your "bad answer" test cases if you want.
    const EXAMPLES = {
      knife: "What kind of knife should I carry for hiking in the bush?",
      water: "How much water should I store for a family of 3 for a 7-day emergency?",
      wound: "How should I treat a deep cut if I am far from medical help?"
    };

    async function callMode(mode) {
      const q = questionEl.value.trim();
      if (!q) {
        throw new Error('Type a question first.');
      }

      const isRag = mode === 'rag';
      const statusEl = isRag ? statusRagEl : statusNoragEl;
      const answerEl = isRag ? answerRagEl : answerNoragEl;
      const metaEl   = isRag ? metaRagEl : metaNoragEl;

      statusEl.textContent = 'Thinking…';
      answerEl.textContent = '';
      metaEl.textContent   = '';

      const res = await fetch('/api/rex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, mode })
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error((isRag ? 'RAG' : 'no-RAG') + ' HTTP ' + res.status + ': ' + txt);
      }

      const data = await res.json();
      answerEl.textContent = data.answer || '(no answer)';
      const modeLabel = data.mode || mode;
      const ms = data.elapsed_ms ?? 0;
      metaEl.textContent = `Mode: ${modeLabel} • Time: ${ms} ms`;
      statusEl.textContent = '';
    }

    async function compare() {
      const q = questionEl.value.trim();
      if (!q) {
        topStatusEl.textContent = 'Type a question first.';
        return;
      }
      topStatusEl.textContent  = 'Running both models…';
      compareBtn.disabled      = true;

      statusRagEl.textContent   = 'Thinking…';
      statusNoragEl.textContent = 'Thinking…';
      answerRagEl.textContent   = '';
      answerNoragEl.textContent = '';
      metaRagEl.textContent     = '';
      metaNoragEl.textContent   = '';

      try {
        await Promise.all([
          callMode('rag').catch(err => { statusRagEl.textContent = 'Error: ' + err.message; }),
          callMode('norag').catch(err => { statusNoragEl.textContent = 'Error: ' + err.message; }),
        ]);
        topStatusEl.textContent = '';
      } catch (e) {
        topStatusEl.textContent = 'One or both calls failed.';
      } finally {
        compareBtn.disabled = false;
      }
    }

    compareBtn.addEventListener('click', () => {
      compare();
    });

    // Ctrl+Enter / Cmd+Enter to compare quickly
    questionEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        compare();
      }
    });

    // Example buttons: set prompt and auto-compare
    exampleButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const key = btn.dataset.example;
        const text = EXAMPLES[key] || "";
        questionEl.value = text;
        questionEl.focus();
        compare();
      });
    });
  </script>
</body>
</html>
"""


def register_web_routes(app, ctx):
    """
    Attach web UI (/) and JSON API (/api/rex) to an existing FastAPI app.

    ctx must provide:
      - stream_ollama (async fn)
      - BM25, DOCS
      - bm25_retrieve
      - MAX_TOTAL_CHARS
      - OLLAMA_MODEL
      - MIN_BM25_SCORE
      - MAX_CONTEXT_DOCS
      - categorize
      - log_interaction
    """

    stream_ollama    = ctx["stream_ollama"]
    BM25             = ctx["BM25"]
    DOCS             = ctx["DOCS"]
    bm25_retrieve    = ctx["bm25_retrieve"]
    MAX_TOTAL_CHARS  = ctx["MAX_TOTAL_CHARS"]
    OLLAMA_MODEL     = ctx["OLLAMA_MODEL"]
    MIN_BM25_SCORE   = ctx["MIN_BM25_SCORE"]
    MAX_CONTEXT_DOCS = ctx["MAX_CONTEXT_DOCS"]
    categorize       = ctx["categorize"]
    log_interaction  = ctx["log_interaction"]

    async def generate_answer_http(user_text: str, mode: str = "rag") -> dict:
        """
        Shared generator for the /api/rex endpoint.
        mode: "rag" (default) or "norag"
        Returns dict with answer, mode_used, elapsed_ms.
        """
        user_text_stripped = (user_text or "").strip()
        if not user_text_stripped:
            raise ValueError("Empty question")

        start_ts = monotonic()
        buffer = ""

        try:
            if mode == "norag":
                concise_sys_prompt = (
                    "You are a helpful, concise assistant replying for a web-based survival helper. "
                    "Keep answers under ~220 words unless the user asks for details. "
                    "Prefer short bullets for steps; avoid long stories."
                )
                async for chunk in stream_ollama(user_text_stripped, sys_prompt=concise_sys_prompt):
                    buffer += chunk
                    if len(buffer) >= MAX_TOTAL_CHARS:
                        buffer = buffer[:MAX_TOTAL_CHARS] + "\n\n…(truncated for length)"
                        break
                mode_used = "norag"
            else:
                if BM25 is not None and DOCS:
                    results = bm25_retrieve(user_text_stripped, BM25, DOCS, top_k=8)
                    best_score = results[0]["score"] if results else 0.0

                    context_chunks = []
                    if results and best_score >= MIN_BM25_SCORE:
                        for hit in results[:MAX_CONTEXT_DOCS]:
                            title = hit.get("title") or ""
                            prefix = f"[{title}] " if title else ""
                            context_chunks.append(prefix + hit["text"])
                    context_block = "\n\n".join(context_chunks)

                    survival_sys_prompt = (
                        "You are Rex, an experienced survival and preparedness instructor talking to a beginner. "
                        "Be concise, practical, and safety-focused. Use short paragraphs or bullets for steps. "
                        "Prefer concrete, real-world advice about bushcraft, homesteading, emergency preparedness, "
                        "medical and safety basics, and gear usage. If something is highly uncertain or depends on "
                        "local regulations, say so explicitly."
                    )

                    if context_block:
                        model_user_prompt = (
                            f"User question:\n{user_text_stripped}\n\n"
                            "Here are relevant excerpts from a survival knowledge base. "
                            "Use them when they clearly help answer the question:\n\n"
                            f"{context_block}\n\n"
                            "Now answer the question above as Rex, using the excerpts plus your own survival expertise. "
                            "Do NOT mention 'documents' or 'context'; just answer normally."
                        )
                    else:
                        model_user_prompt = (
                            f"The user asked:\n{user_text_stripped}\n\n"
                            "Answer as Rex the survival instructor using your general survival knowledge. "
                            "If something is uncertain or risky, explain the tradeoffs and safety precautions."
                        )

                    async for chunk in stream_ollama(model_user_prompt, sys_prompt=survival_sys_prompt):
                        buffer += chunk
                        if len(buffer) >= MAX_TOTAL_CHARS:
                            buffer = buffer[:MAX_TOTAL_CHARS] + "\n\n…(truncated for length)"
                            break

                    mode_used = "rag"
                else:
                    concise_sys_prompt = (
                        "You are a helpful, concise assistant replying for a web-based survival helper. "
                        "Keep answers under ~220 words unless the user asks for details. "
                        "Prefer short bullets for steps; avoid long stories."
                    )
                    async for chunk in stream_ollama(user_text_stripped, sys_prompt=concise_sys_prompt):
                        buffer += chunk
                        if len(buffer) >= MAX_TOTAL_CHARS:
                            buffer = buffer[:MAX_TOTAL_CHARS] + "\n\n…(truncated for length)"
                            break
                    mode_used = "fallback"

            elapsed_ms = int((monotonic() - start_ts) * 1000)

            try:
                log_interaction(
                    user_id=0,
                    message=user_text_stripped,
                    reply_len=len(buffer or ""),
                    response_time_ms=elapsed_ms,
                    category=categorize(user_text_stripped or ""),
                    error=False,
                    meta={"model": OLLAMA_MODEL, "mode": f"web_{mode_used}"},
                )
            except Exception:
                pass

            return {"answer": buffer or "No response.", "mode": mode_used, "elapsed_ms": elapsed_ms}

        except Exception as e:
            elapsed_ms = int((monotonic() - start_ts) * 1000)
            try:
                log_interaction(
                    user_id=0,
                    message=user_text_stripped,
                    reply_len=len(buffer or ""),
                    response_time_ms=elapsed_ms,
                    category=categorize(user_text_stripped or ""),
                    error=True,
                    meta={"model": OLLAMA_MODEL, "mode": f"web_{mode}", "exception": str(e)},
                )
            except Exception:
                pass
            raise

    async def rex_page(request: Request):
        return HTMLResponse(content=REX_HTML)

    async def api_rex(request: Request):
        payload = await request.json()
        question = (payload.get("question") or "").strip()
        mode = (payload.get("mode") or "rag").strip().lower()
        if mode not in ("rag", "norag"):
            mode = "rag"
        if not question:
            raise HTTPException(status_code=400, detail="Missing 'question'")

        try:
            result = await generate_answer_http(question, mode=mode)
            return JSONResponse(result)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Register routes on the given app
    app.add_api_route("/", rex_page, methods=["GET"])
    app.add_api_route("/api/rex", api_rex, methods=["POST"])

