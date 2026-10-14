# OmniFlow AI - implementation status

Legend: **Verified** (tested and passing) · **Verified locally** (real local model, no API) ·
**Verified live** (also checked against the real service) · **Not tested** · **Planned**.

_Last updated: 2026-10-13, External Integrations & Multi-Tool Orchestration milestone._

## Phase 1: Foundation

| Item | Status | Evidence |
|---|---|---|
| Python 3.12 project `.venv`, clean dependencies | Verified | `pip check`: no broken requirements |
| Typed config, `.env` loading, `SecretStr` | Verified | `tests/test_config.py` |
| Secret-redacting structured logging | Verified | `tests/test_logging.py` |
| Safe error classification | Verified | `tests/test_errors.py` |
| Gemini model factory + fallback | Verified live | `tests/test_llm.py`; user ran `diagnostics --live` |
| No hardcoded secrets; sensitive paths gitignored | Verified | `tests/test_security.py` |
| Tutorial fixes (Alpha Vantage key, timeouts, LangChain 1.x import) | Verified | `tests/test_tutorial_fixes.py` |

## Phase 2: Agent core

| Item | Status | Evidence |
|---|---|---|
| Typed per-turn state (route, plan, tool results, steps, verification) | Verified | `tests/test_agent.py` |
| Deterministic routes: `/help`, `/search`, secret-request refusal, input validation | Verified (no model call) | `tests/test_agent.py` |
| Model planner: chat / tool / clarify / unsupported | Verified with fakes; **verified live** for the tool route | `tests/test_agent.py`; live probe 2026-10-10 |
| Safe fallback when the model's plan is unparseable or invalid | Verified | `tests/test_agent.py` |
| Central tool registry (allowlist, schemas, permissions, availability) | Verified | `tests/test_registry_executor.py` |
| Controlled execution (validation, timeouts, bounded retries, output limits) | Verified | `tests/test_registry_executor.py` |
| Tool-call limit, step limit, deduplication, acyclic graph | Verified | `tests/test_agent.py` |
| Verification and abstention (citations, links, images, failures, no results) | Verified | `tests/test_agent.py` |
| Untrusted-content handling (evidence quarantine, Markdown escaping) | Verified | `tests/test_agent.py` |
| Human-approval foundation (interrupt, exact-fingerprint approval) | Verified with a test-only tool | `tests/test_agent.py`, `tests/test_app.py` |
| `web_search` (DuckDuckGo via `ddgs`) | Verified with mocks; **verified live** | `tests/test_web_search.py`; live probe; opt-in `tests/test_live_web_search.py` |
| Streamlit: route, executed tools, verification, evidence, approval card | Verified | `tests/test_app.py` (AppTest); headless start HTTP 200 |

(Phase 2 note) The approval flow was first proven with test tools; today it guards image
generation and other approved actions.

## Phase 3: Multi-PDF RAG

| Item | Status | Evidence |
|---|---|---|
| Multi-file PDF upload page (limits on type, size, count, pages, text) | Verified (logic + page rendering) | `tests/test_documents.py`, `tests/test_app.py`. The upload widget itself can't be driven by AppTest, so **upload clicks are untested** |
| Page-by-page extraction keeping file name and page number | Verified | `tests/test_documents.py` |
| Empty, malformed, encrypted, image-only, oversized, wrong-type files | Verified (rejected with clear messages) | `tests/test_documents.py` |
| Chunking within pages (configurable size and overlap) | Verified | `tests/test_documents.py` |
| Local embeddings (`bge-small-en-v1.5` via fastembed) | **Verified locally** | Smoke test: 4/4 queries found the correct file and page; +228 MB RAM |
| Relevance threshold (no-match detection) | **Verified locally** | Calibrated to 0.55 (relevant 0.71-0.85, unrelated 0.38-0.47) |
| Per-session index, isolation, clear one session, remove one document, idle expiry | Verified | `tests/test_documents.py`, `tests/test_document_agent.py` |
| No re-indexing of an identical PDF in a session | Verified | `tests/test_documents.py` |
| `document_search` tool through the existing registry, executor and verification | Verified with fakes; **verified live** | `tests/test_document_agent.py`; live probe (3 Gemini requests) |
| Page citations (`file.pdf, page N`) across multiple PDFs | Verified; **verified live** | as above |
| Insufficient evidence vs. no match vs. tool failure | Verified with fakes; no-match **verified live** | `tests/test_document_agent.py`; live probe |
| Prompt-injection text in PDFs | Verified with fakes; **one live case verified** (injected commands ignored) | `tests/test_document_agent.py`; live probe |
| Model cannot choose another session's documents | Verified | `tests/test_document_agent.py` |
| Documents-only mode and `/docs` command | Verified | `tests/test_document_agent.py`, `tests/test_app.py` |
| OCR for scanned pages (Tesseract) | **Not tested with real Tesseract** (not installed); logic verified with a fake OCR engine | `tests/test_documents.py` |
| Persistence | In memory only (by design) | Documents are lost on clear, after 120 min idle, or on restart |

## Phase 4: Data analysis and SQLite SQL agent

| Item | Status | Evidence |
|---|---|---|
| CSV and XLSX ingestion (multiple files, multiple sheets) | **Verified locally** | `tests/test_data.py` (real pandas/openpyxl) |
| Rejection of bad files: wrong type, `.xlsm`/`.xls`, empty, binary, malformed, oversized, too many rows/columns, zip bombs, macros | **Verified locally** | `tests/test_data.py` |
| Duplicate/blank column names, encoding fallback, non-numeric values, malformed CSV lines | **Verified locally** | `tests/test_data.py` |
| Profiling: shape, types, missing, duplicates, stats, outliers, top values, dates, correlations, identifier detection | **Verified locally** | `tests/test_data.py` |
| Allowlisted operations (10) with validated parameters and filters; hand-checked expected values | **Verified locally** | `tests/test_data.py` |
| Charts (bar, line, histogram, scatter with sampling, correlation heatmap) | **Verified locally** (Vega-Lite specs built) | `tests/test_data.py` |
| No code execution: unknown operations or extra args (e.g. `code`) rejected; code-like column names treated as data | **Verified** | `tests/test_data.py`, `tests/test_data_sql_agent.py` |
| Formula-injection-safe CSV export | **Verified** | `tests/test_data.py` |
| Dataset session isolation and scoped clearing | **Verified** | `tests/test_data.py`, `tests/test_data_sql_agent.py` |
| SQL layer 1: parser validation (single read-only query, no recursive CTEs, table and function allowlists) | **Verified locally** | `tests/test_sql.py` (26 attack/invalid cases) |
| SQL layer 2: read-only connection, query-only setting, ATTACH disabled, authorizer, timeout, row limit | **Verified locally**, including attacks that bypass layer 1 | `tests/test_sql.py` |
| Database unchanged after blocked queries (file hash, row counts, tables) | **Verified locally** | `tests/test_sql.py` |
| Explicitly configured databases only; names, never paths; your DB needs a table allowlist | **Verified** | `tests/test_sql.py` |
| Agent routing: data, SQL, documents and web; focused modes; clarification | Verified with a scripted model | `tests/test_data_sql_agent.py` |
| Numeric cross-check of answers against computed results | **Verified** | `tests/test_data_sql_agent.py` |
| Instructions inside spreadsheet cells treated as untrusted data | Verified with a scripted model | `tests/test_data_sql_agent.py` |
| Data Analysis and SQL pages | **Verified** (AppTest) except the upload widget, which AppTest can't drive | `tests/test_app.py` |
| Data Analysis: many consecutive questions; the input stays pinned (successes, tool failures, model errors) | **Verified** (AppTest) and **in a real browser** (Edge, 4 questions) | `tests/test_app.py`; Playwright check |
| Import CSV/XLSX into a session-only SQLite database (explicit button; safe names; parameterised inserts) | **Verified locally**, including a real upload in Edge | `tests/test_sql_import.py`, `tests/test_app.py` |
| Imported tables: read-only queries, both safety layers, isolation, selected-database restriction, cleanup | **Verified** | `tests/test_sql_import.py` (35 tests) |
| Live Gemini routing to data/SQL tools | **Verified live** (4 requests): Gemini chose `group_aggregate` and wrote a valid 3-table join; both answers grounded | live probe 2026-10-11 |

### Phase 4 follow-up fix
| Item | Status | Evidence |
|---|---|---|
| Arrow serialisation warning for mixed-type columns (e.g. Excel pivot "Column Labels") | **Fixed and verified**: display-only conversion of mixed columns; stored data unchanged | `tests/test_app.py` (reproduces the exact error without the fix) |

## Phase 5: Web, news and academic research

| Item | Status | Evidence |
|---|---|---|
| Locked-down fetcher (HTTPS, host allowlist, no private/metadata/CGNAT addresses incl. NAT64, redirect checks, size cap, timeout) | **Verified** | `tests/test_research_news.py` |
| `news_search`: DuckDuckGo News + 5 official RSS feeds; dedupe, date window, publication vs retrieval date, source balance, partial-coverage notes | **Verified** with mocks; **verified live** (2026-10-12) | `tests/test_research_news.py`, `tests/test_live_research.py` |
| RSS/Atom parsing with defusedxml (entity bombs, malformed XML, missing dates, bad links) | **Verified** | `tests/test_research_news.py` |
| `academic_search`: OpenAlex + arXiv, shared schema, DOI/title dedupe, relevance/recency, missing metadata left empty, arXiv 3 s throttle | **Verified** with mocks; **verified live** | `tests/test_research_academic.py`, `tests/test_live_research.py` |
| `web_search` results include the domain | **Verified** | `tests/test_research_agent.py` |
| Agent: tool choice, multi-tool, `/news` and `/papers`, citations with dates/DOIs, abstract-based labelling, abstention, prompt-injection quarantine | **Verified** with a scripted model; **verified live** (Gemini chose `news_search` / `academic_search`; answers grounded) | `tests/test_research_agent.py`; live probe |
| Assistant UI: news and paper evidence panels (publisher, published/retrieved dates, authors, venue, DOI) | **Verified** (AppTest) | `tests/test_app.py` |
| Opt-in live tests can actually run (`OMNIFLOW_RUN_LIVE_TESTS=1`) | **Fixed**: the test isolation used to clear this switch | `tests/conftest.py` |

## Phase 6: Saved conversations, streaming, image understanding

| Item | Status | Evidence |
|---|---|---|
| Root cause of "New chat loses the conversation": `new_chat()` replaced the thread id and emptied the in-session message list, the agent used `InMemorySaver`, and nothing listed old threads | **Fixed** | `omniflow/ui/state.py` |
| Conversation store (SQLite, parameterised, versioned schema, exactly-once per turn and role, secret redaction, cascade delete) | **Verified** | `tests/test_conversation_store.py` |
| Agent checkpoints in SQLite (`SqliteSaver`): memory survives a restart; deleting a conversation deletes its checkpoints | **Verified** | `tests/test_streaming.py`, `tests/test_history_app.py` |
| Sidebar: New chat keeps old conversations; open, rename, delete with confirmation; refresh reopens via `?c=`; restart keeps everything | **Verified** (AppTest) | `tests/test_history_app.py` |
| Interrupted turns: partial reply saved once and marked; unanswered question marked; pending approval restored after restart | **Verified** (AppTest) | `tests/test_history_app.py` |
| Genuine streaming: the first token reaches the UI while the model is still generating | **Verified** (gated fake model); **verified live** 2026-10-13: 6 chunks arrived between 1.84 s and 2.57 s, and the stream equalled the final answer | `tests/test_streaming.py`; live probe |
| Image validation: content-based format check (JPEG/PNG/WebP), size and pixel limits before decoding, truncated/non-image/spoofed files refused, EXIF/GPS stripped, orientation applied, downscaled | **Verified** | `tests/test_images.py` |
| `image_analyze`: only this conversation's images, model cannot pick a conversation, per-call limit, safe errors, untrusted image text, cited as "may contain errors" | **Verified** with a fake vision model; **verified live** 2026-10-13 (`gemini-3.5-flash-lite` read "OMNIFLOW 42" and two shapes correctly, 4.3 s) | `tests/test_images.py`, `tests/test_history_app.py`; live probe |
| PyArrow mixed-type display fix (Phase 4) still in place | **Verified** | `tests/test_app.py` |

## Memory & Preference Manager + Streamlit Cloud readiness

| Item | Status | Evidence |
|---|---|---|
| Memory store: separate SQLite file, versioned schema (atomic migration), parameterised SQL, owners, categories, scopes, source, active flag, optional expiry | **Verified** | `tests/test_memory_store.py` |
| Secrets refused (13 credential shapes + registered secrets), personal data needs confirmation, exact and near duplicates blocked (including across concurrent sessions) | **Verified** | `tests/test_memory_policy.py`, `tests/test_memory_store.py` |
| Commands (remember / remember for project / forget with confirmation / show) handled without a model call; preference statements only suggested | **Verified** (AppTest) | `tests/test_memory_app.py` |
| Retrieval: deterministic, scope-aware, at most 8 items and 1,500 characters | **Verified** | `tests/test_memory_policy.py`, `tests/test_memory_app.py` |
| Memories quarantined in prompts (quoted JSON, closing tag neutralised); cannot approve tools or bypass the secrets refusal; never cited as evidence | **Verified** | `tests/test_memory_agent.py` |
| Memory page: filter, add, edit, disable or enable, delete with confirmation, clear all after typing DELETE, on/off switch, active project; success shown only after the write | **Verified** (AppTest) | `tests/test_memory_app.py` |
| Memory deletion never touches conversations, checkpoints or files | **Verified** | `tests/test_memory_store.py` |
| Restart persistence (local) | **Verified** | `tests/test_memory_store.py`, `tests/test_memory_app.py` |
| Owner scoping for conversations (schema v2, Phase 6 databases migrated in place) and memories; `per_session` mode on Community Cloud | **Verified locally** (simulated sessions) | `tests/test_memory_app.py`, `tests/test_deployment.py` |
| Failure handling: unwritable, damaged, newer-version or failed-migration databases are reported; chat keeps working; no silent in-memory fallback | **Verified** | `tests/test_memory_store.py`, `tests/test_memory_app.py`, `tests/test_deployment.py` |
| Logs carry memory ids and categories only, never memory text | **Verified** | `tests/test_deployment.py` |
| Startup without network or credentials; relative storage paths independent of the working directory; no machine-specific paths in the source | **Verified locally** | `tests/test_deployment.py` |
| `requirements.txt` resolves to Linux CPython 3.12 wheels (102 packages, no source builds) | **Verified locally** (pip dry run against PyPI) | see README "Deploying" |
| Headless start from a clean working directory with no `.env` | **Verified locally** (`/_stcore/health` = ok) | manual check |
| Running on Streamlit Community Cloud | **Not verified** - not deployed | remaining hosted test listed in README |
| Durable hosted memory | **Not available** - needs an external database, credential and sign-in (not approved) | `docs/API_KEYS_AND_SERVICES.md` |

## Coding assistant milestone

| Item | Status | Evidence |
|---|---|---|
| Project intake: ZIP (traversal, absolute, drive, symlink, encrypted, bomb, size and count limits) and local folders (no roots or home, symlinks skipped, app storage excluded) | **Verified** (7/7 hostile archives blocked) | `tests/test_code_workspace.py` (symlink test skipped on Windows: needs privileges) |
| File policy: `.env`, keys, credentials, `.git`, venvs, caches, node_modules, databases, binaries never indexed; secret values redacted with line numbers preserved | **Verified** | `tests/test_code_workspace.py`, `python -m omniflow.code.evaluate` |
| Index, lexical and error-aware search, outline, entry points, static name tracing, bounded reads | **Verified**: search top-1 6/6 and top-3 6/6 on the synthetic questions | `tests/test_code_workspace.py`, `tests/test_code_eval.py` |
| Read-only agent tools (search, read, outline, trace, security scan, propose patch) in a focused `code` mode; project and owner from trusted context; repository text quarantined as evidence; `file, lines` citations | **Verified** with a scripted model | `tests/test_code_agent.py` |
| Deterministic security review with severity, confidence, redacted evidence, impact and fix | **Verified**: precision 1.0, recall 1.0 on a 26-snippet labelled corpus (written alongside the rules) | `tests/test_code_security.py` |
| Patches: exact find/replace to unified diff, pasted-diff review, syntax / security / secret checks, review-only by default; approved apply with change detection, backup, atomic write and undo; CRLF and BOM preserved | **Verified** | `tests/test_code_patches.py`, `tests/test_code_app.py` |
| Checks: allowlist, approval fingerprint, trusted-only pytest, scrubbed environment, timeout kills the process tree, output cap and redaction, one at a time | **Verified** (real subprocesses on this machine) | `tests/test_code_checks.py` |
| Code page: open, overview, five actions, diff viewer, approval ticks, hosted mode hides local folders | **Verified** (AppTest) | `tests/test_code_app.py` |
| Live Gemini (`gemini-3.5-flash-lite`) on the fixture project | **Run once, opt-in**: repository question cited `app/orders.py` (pass), README prompt injection ignored (pass), bug diagnosis failed first and passed after error-aware search was added | `tests/test_live_code.py` |
| Hosted deployment of coding features | **Not verified**. Local folders, execution and folder patching are off when `per_session` | README |

## Smart Task & Project Planner milestone

| Item | Status | Evidence |
|---|---|---|
| Plan proposals: typed schema (keys, priorities, ISO dates, date source, dependencies, one level of subtasks, cycle detection, limits); kept as unsaved proposals | **Verified** | `tests/test_planner_core.py`, `tests/test_planner_agent.py` |
| Persistent projects: separate SQLite file, versioned schema, FKs and cascades, owner scoping, one-transaction writes, idempotent tokens, failure handling (unwritable, damaged, newer, failed write) | **Verified** | `tests/test_planner_core.py` |
| Change control: typed operations, preview with before/after and impact, stale-preview rejection, significant-change confirmation, no self-approval (no writing tools) | **Verified** | `tests/test_planner_core.py`, `tests/test_planner_agent.py`, `tests/test_planner_app.py` |
| Deterministic ranking (documented rule, exact scores, urgency bands, ties) and progress (leaf tasks, cancelled excluded, effort-weighted, unknown when empty) | **Verified** | `tests/test_planner_core.py` |
| Overdue and upcoming only for confirmed deadlines; suggested dates kept apart | **Verified** | `tests/test_planner_core.py` |
| Agent integration: focused `planner` mode, proposals from the Assistant too, computed evidence with the numeric cross-check, memories as untrusted context, today's date in the prompt | **Verified** with a scripted model | `tests/test_planner_agent.py` |
| Planner page: plan form, editable proposal, explicit save, board, timeline, task editing, milestones, Ask, archive and delete with confirmation, per-session privacy | **Verified** (AppTest) | `tests/test_planner_app.py` |
| Live Gemini (`gemini-3.5-flash-lite`) plan proposals | **Run, opt-in**: the first run failed (Gemini sent `scope` and `risks` as objects, so the plan was rejected); after normalisation, both live tests passed (no invented deadline; dates within the user's deadline) | `tests/test_live_planner.py` |
| Reminders, calendar, background jobs | **Not implemented** (out of scope) | - |

## Data Storytelling & Forecasting milestone

| Item | Status | Evidence |
|---|---|---|
| Series preparation: explicit columns, identifier refusal, date parsing, duplicates, sum/mean, frequency inference, partial-period trimming, gap policy | **Verified** | `tests/test_data_forecast.py` |
| Trends: changes (zero base), OLS slope, Mann–Kendall (checked by hand), robust anomalies (planted spike found; busier periods not flagged), seasonality only with 2 cycles | **Verified** | `tests/test_data_forecast.py` |
| Forecasting: 6 candidates, rolling-origin holdout without leakage (spy test), metrics checked by hand, strict baseline rule, intervals plus measured coverage, input validation | **Verified** | `tests/test_data_forecast.py` |
| Deterministic evaluation (20 seeds per scenario) | **Measured**: trend direction 1.0; anomaly precision and recall 1.0; seasonal-naive chosen 20/20; trend methods beat naive 20/20; random walks kept naive 15/20; 95% interval coverage 0.95 | `python -m omniflow.data.evaluate`, `tests/test_data_evaluate.py` |
| Data story and recommendations: numbers equal pandas, every recommendation cites real findings, confidence labels, no outcome promises, correlation labelled as association | **Verified** | `tests/test_data_story.py` |
| Agent tools `dataset_story` / `dataset_trends` / `dataset_forecast`: typed schemas, plain-Python results (fixes a checkpoint serialisation crash found here), bounded evidence, numeric cross-check, prompt injection in cells and column names stays data | **Verified** with a scripted model | `tests/test_data_story.py` |
| Data page section (story, trends, outliers, forecast, recommendations, JSON export) | **Verified** (AppTest) | `tests/test_data_story.py` |
| Live Gemini | **Partly run (opt-in)**: the story test passed (grounded on "North", no causal wording; verification flagged one number not in the results). The forecast test **did not complete**: Google returned 503 "model experiencing high demand" twice | `tests/test_live_data_story.py` |

## Integrations milestone (revised 2026-10-13: image generation only)

| Item | Status | Evidence |
|---|---|---|
| Root cause of "cat" request returning a MOCK placeholder: the leftover `OMNIFLOW_INTEGRATIONS_MODE=mock` in `.env` overrode `OMNIFLOW_IMAGE_PROVIDER=cloudflare` in `IntegrationsHub.image_info()` | **Fixed**: the provider is now chosen only by `OMNIFLOW_IMAGE_PROVIDER`; the old setting is ignored | `tests/test_integrations.py`, `tests/test_integration_routing.py` |
| Cloudflare client: documented endpoint, Bearer token, base64 decode, image validation, specific errors (401/403/404/429/timeout) with Cloudflare error codes, never retried, no mock fallback; account ids masked in logs | **Verified offline** (mocked HTTP) | `tests/test_integrations.py`, `tests/test_orchestration.py` |
| Live Cloudflare generation | **Failed (not working yet)**: one bounded call on 2026-10-13 returned **HTTP 401** (`application/json`), i.e. Cloudflare rejected the API token. No image was generated | manual diagnostic |
| Approval, rejection (nothing created), mock labelling, image display in chat | **Verified** (AppTest) | `tests/test_integration_routing.py` |
| Gmail and Google Calendar (OAuth, tokens, tools, UI) | **Removed** at the user's request | `tests/test_integrations.py` (modules absent), `tests/test_integration_routing.py` |
| Multi-round orchestration; Document → Planner proposal chain | **Verified** with a scripted model | `tests/test_orchestration.py` |

## External integrations - Milestone 1: Tavily Research Agent (2026-10-14)

| Item | Status | Evidence |
|---|---|---|
| `TAVILY_API_KEY` in typed settings (`SecretStr`, optional, placeholders ignored); app runs without it | **Verified** | `tests/test_tavily_search.py` |
| `tavily_search` tool: official endpoint, Bearer key, basic depth, no generated answer/raw content/images; input validation; result count and text bounds; score validation; tolerant parsing | **Verified** with mocked HTTP | `tests/test_tavily_search.py` |
| Errors: missing key, 401, 400/422, 429, 432/433, 5xx, timeout, network, non-JSON, wrong shape, empty results; only transient errors retried once; key and response body never in messages or logs | **Verified** with mocked HTTP | `tests/test_tavily_search.py` |
| Registration, planner offering + guidance, `/research` command, unchanged greeting / arithmetic / `/search` / `/help` routes | **Verified** with a scripted model | `tests/test_tavily_search.py`, existing routing suites |
| Prompt-injection quarantine, failure without crash or invented answer, titles and URLs in evidence and Sources | **Verified** with a scripted model | `tests/test_tavily_search.py` |
| Live Tavily API | **Verified live** once (HTTP 200, 3 results, 1 credit) | manual probe through the tool |
| Live end-to-end research turn with Gemini choosing `tavily_search` | **Not verified** (the planner's choice between `tavily_search` and `web_search` is model-dependent) | - |

## External integrations - Milestone 2: Open-Meteo Weather Agent (2026-10-14)

| Item | Status | Evidence |
|---|---|---|
| `weather_lookup` tool: keyless, read-only; geocoding then forecast; current conditions + 1-16 day daily forecast; metric/imperial; `OMNIFLOW_WEATHER_ENABLED` switch | **Verified** with mocked HTTP | `tests/test_weather.py` |
| Location resolution: real geocoded coordinates only; region/country qualifiers and `country_code`; ambiguous names produce a question; clearly dominant match used and alternatives disclosed; no-match message | **Verified** with mocked HTTP | `tests/test_weather.py` |
| Local timestamps and dates with timezone name, abbreviation and UTC offset; requested-day bound; missing or invalid values reported as "not provided" | **Verified** with mocked HTTP | `tests/test_weather.py` |
| Errors: timeout/network/5xx retried once; 429 and 4xx not retried; malformed JSON and shapes; fixed HTTPS hosts via `safe_get` (now also reports the HTTP status) | **Verified** with mocked HTTP | `tests/test_weather.py` |
| Routing: planner guidance, `/weather`, `/help`; greeting, arithmetic, Tavily `/research` unchanged; injection quarantined; outage and ambiguity don't crash or invent; Open-Meteo attribution in Sources; invented numbers flagged | **Verified** with a scripted model | `tests/test_weather.py` |
| Live Open-Meteo API | **Verified live** once ("Berlin, Germany", 2 days: 2 calls) | manual probe through the tool |
| Live end-to-end weather turn with Gemini choosing `weather_lookup` | **Not verified** | - |

## External integrations - Milestone 3: Frankfurter currency tool (2026-10-14)

| Item | Status | Evidence |
|---|---|---|
| `currency_convert` tool: keyless, read-only; latest or historical (`YYYY-MM-DD`) rate; optional amount conversion in exact `Decimal` arithmetic; display rounding only; `OMNIFLOW_CURRENCY_ENABLED` switch | **Verified** with mocked HTTP | `tests/test_currency.py` |
| Input checks inside the tool (so problems become questions, not a fallback answer from memory): missing, ambiguous, invalid, same currency; amount (positive, numeric, finite, at most 1 trillion, at most 8 decimals); date (format, real date, not future, not before 1948) | **Verified** | `tests/test_currency.py` |
| Errors: 404 = no published rate (not replaced by another date), 400/422 = unsupported code, other 4xx and 429 not retried; 5xx, timeout and network retried once; malformed JSON, schema, rate, date or pair rejected; fixed HTTPS host via `safe_get` | **Verified** with mocked HTTP | `tests/test_currency.py` |
| Evidence: Frankfurter URL with the reference date in Sources and the evidence panel; reference-rate disclaimer; invented amounts flagged by the verifier | **Verified** with a scripted model | `tests/test_currency.py` |
| Routing: planner guidance and tool binding; greeting, arithmetic, `/weather` unchanged; `/help` example | **Verified** with a scripted model | `tests/test_currency.py`, existing routing suites |
| Live Frankfurter API | **Verified live** once through the tool (1000 INR to USD: rate dated 2026-10-13, 1.5 s). Schema and 404/422 behaviour were confirmed beforehand with 7 read-only requests | manual smoke test |
| Live end-to-end currency turn with Gemini choosing `currency_convert` | **Not verified** | - |

## Later phases

| Capability | Status |
|---|---|
| Audio and video understanding | Not scheduled (excluded from Phase 6) |
| Code assistant | Implemented (Code page) |
| Gmail, Google Calendar | Removed (2026-10-13, at the user's request) |
| Image generation | Implemented (Cloudflare Workers AI; editing not supported) |
| Workflows, evaluation, docs, UI polish | Planned (Phase 10) |
| GitHub Actions CI, Docker | Planned (Phase 11) |
| Deployment | Planned (Phase 12, approval-gated) |
| Persistent conversation history | Planned (with SQLite work) |
