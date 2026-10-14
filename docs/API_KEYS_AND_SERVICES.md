# API keys and external services

The single source of truth for every credential and external service OmniFlow AI uses.
It is updated whenever an integration is added or changed.

- **Last reviewed:** 2026-10-13 (Memory & Preference Manager + Streamlit Cloud readiness)
- **Project rule: free only.** No paid model, paid API or billing is used, and none is used as a
  silent fallback. Anything whose free status can't be verified is not added.
- **Rule:** secrets live only in your local `.env` (gitignored). `.env.example` holds names only.
- **Status meanings:** *not needed* · *pending* (you have not configured it yet) ·
  *configured* (the variable is set; not proven to work) · *verified* (a live check passed).
  A key that is only present in `.env` is never reported as verified.

## Inventory

| Feature | Provider | Credential | Required? | Free tier (as of 2026-10-09) | Billing risk | Env variable | Status |
|---|---|---|---|---|---|---|---|
| Chat, planning, answer synthesis | Google Gemini API | API key | **Required** for chat | Yes. `gemini-3.8-flash` and `gemini-3.5-flash-lite` are "Free of charge" on the free tier. Rate limits are per project and only shown in AI Studio | None unless you enable billing on the key's Cloud project yourself | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | **Verified** - live diagnostics passed (user, 2026-10-09). Live agent turn with `gemini-3.5-flash-lite` passed (2026-10-10) |
| Image understanding (`image_analyze`) | Google Gemini API (same key and model as chat: `GEMINI_MODEL`) | **None new** - reuses `GEMINI_API_KEY` | Optional (needs an attached image) | Yes. Google's pricing page (checked 2026-10-13) lists `gemini-3.5-flash-lite` free-tier input as "Free of charge"; image input is counted as tokens (about 258 per 768×768 tile) within the free-tier rate limits | None unless you enable billing yourself | `OMNIFLOW_IMAGES_ENABLED`, `OMNIFLOW_IMAGE_*` limits (not secrets) | **Verified live** 2026-10-13 - one test image read correctly |
| Saved conversations | **Local**: SQLite files in `data/` (Python `sqlite3` + `langgraph-checkpoint-sqlite` 3.1.1, MIT) | **None** | On by default | Local files, no service | None | `OMNIFLOW_HISTORY_ENABLED`, `OMNIFLOW_CONVERSATIONS_DB`, `OMNIFLOW_CHECKPOINTS_DB` (paths, not secrets) | **Verified locally** |
| Long-term memory (preferences, project facts) | **Local**: SQLite file `data/omniflow_memory.db` (Python `sqlite3`) | **None** | On by default | Local file, no service | None | `OMNIFLOW_MEMORY_ENABLED`, `OMNIFLOW_MEMORY_DB`, `OMNIFLOW_ACCESS_MODE` (not secrets) | **Verified locally** |
| Image generation (optional) | Cloudflare Workers AI, `@cf/black-forest-labs/flux-1-schnell` | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, plus `OMNIFLOW_IMAGE_PROVIDER=cloudflare` | Optional | Free plan: 10,000 Neurons/day, resets 00:00 UTC; on the Free plan requests beyond that fail rather than bill (checked 2026-10-13). Model terms: Black Forest Labs | None on the Free plan | as listed | **Configured by you; first live call failed with HTTP 401** (token rejected) - see below |
| Data storytelling & forecasting | **Local**: pandas and NumPy (already installed); Gemini (existing key) only for explanations | **None new** | Optional | Free; no forecasting service, no model download | None | `OMNIFLOW_DATA_FORECAST_MAX_HORIZON`, `OMNIFLOW_DATA_MAX_SERIES_PERIODS` (limits) | **Verified locally**; live explanation partly run |
| Planner (Planner page) | **Local**: SQLite file `data/omniflow_planner.db`; Altair (already installed) for the timeline; Gemini (existing key) for proposals and explanations | **None new** | Optional | Local storage is free; proposals use the existing free Gemini quota | None | `OMNIFLOW_PLANNER_ENABLED`, `OMNIFLOW_PLANNER_DB` (not secrets) | **Verified locally**; live proposals run (opt-in) |
| Coding assistant (Code page) | **Local**: Python `ast`, `zipfile`, `difflib`; optional Ruff (dev dependency) for lint checks; Gemini (existing key) for explanations | **None new** | Optional | Local analysis is free; explanations use the existing free Gemini quota | None | `OMNIFLOW_CODE_*` (limits, switches, trusted roots - not secrets) | **Verified locally**; live Gemini evaluation run once (opt-in) |
| PDF retrieval embeddings (`document_search`) | **Local**: `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX Runtime, CPU) | **None** | Required for document Q&A; downloaded once automatically | Free: open model (MIT), runs on your machine. One-time download (~67 MB) from Hugging Face, which needs no account or token | **None.** No API is called | `OMNIFLOW_EMBEDDING_MODEL` (model choice, not a secret) | **Verified locally** - real model, synthetic PDFs, correct file and page for 4/4 queries (2026-10-10) |
| OCR for scanned pages (optional) | **Local**: Tesseract OCR (UB-Mannheim Windows build) | **None** | Optional | Free, open source (Apache-2.0). You install it yourself; OmniFlow never installs it | None | `OMNIFLOW_OCR_ENABLED`, `OMNIFLOW_TESSERACT_CMD` (paths, not secrets) | **Not installed and not tested** with real Tesseract; logic tested with a fake OCR engine |
| Data analysis (`dataset_profile`, `dataset_analyze`, `dataset_chart`) | **Local**: pandas 3.0.6, NumPy 2.5.3, Altair 6.3.0 (charts drawn in your browser), openpyxl 3.1.5 + defusedxml 0.7.1 (.xlsx) | **None** | Optional (needs an upload) | Open-source libraries running on your machine. No API and no account | **None** (only local CPU and electricity) | `OMNIFLOW_DATA_*` limits (not secrets) | **Verified locally** - real pandas on synthetic CSV/XLSX (tests + smoke test) |
| SQL agent (`sql_schema`, `sql_query`) | **Local**: Python's built-in `sqlite3` (SQLite 3.42.0) + sqlglot 30.22.0 (SQL parser) | **None** | Optional | Local file database, no server. A synthetic **sample** database is generated locally | **None** | `OMNIFLOW_SQL_*` (paths and allowlists, not secrets) | **Verified locally** - real SQLite (tests + smoke test) |
| News & trends (`news_search`) | DuckDuckGo News (via `ddgs`) + official RSS feeds: WHO, NASA, Reserve Bank of India, European Commission, arXiv cs.AI | **None** | Optional (on by default) | No account. DuckDuckGo News is an unofficial scraper and was observed throttling repeated automated requests (empty results). Feeds are public institutional RSS | None | `OMNIFLOW_NEWS_ENABLED` (on/off) | **Verified live** 2026-10-12: dated DuckDuckGo News results, all 5 feeds reachable; also later observed DuckDuckGo News returning nothing (throttling) |
| Academic research (`academic_search`) | OpenAlex API + arXiv API | **None** (OpenAlex works keyless) | Optional (on by default) | OpenAlex: free; keyless budget about 1,000 credits/day ≈ **100 searches/day** (a search costs 10 credits). arXiv: free; terms ask for **at most 1 request every 3 s** (enforced) | None (OpenAlex sells higher limits; OmniFlow never uses paid plans) | `OMNIFLOW_ACADEMIC_ENABLED` (on/off) | **Verified live** 2026-10-12: real papers from both providers |
| Web search (`web_search` tool) | DuckDuckGo, via the `ddgs` library | **None** (keyless) | Optional (on by default) | No account and no API. `ddgs` scrapes DuckDuckGo's public results pages, so it's unofficial, can be rate-limited, and can break if the pages change | None | `OMNIFLOW_WEB_SEARCH_ENABLED` (on/off, not a secret) | **Verified** - one live search returned 5 results (2026-10-10) |
| Research search (`tavily_search` tool) | Tavily Search API (`POST https://api.tavily.com/search`, official) | `TAVILY_API_KEY` (your own) | Optional - without it the tool is unavailable and `web_search` is used | Free Researcher plan: 1,000 API credits/month, no credit card; a basic search costs 1 credit; development keys 100 requests/minute (docs.tavily.com, checked 2026-10-14) | None on the free plan; OmniFlow never enables pay-as-you-go | `TAVILY_API_KEY` (secret), `OMNIFLOW_TAVILY_TIMEOUT_SECONDS`, `OMNIFLOW_TAVILY_MAX_RESULTS` | **Verified live** 2026-10-14: one request through `tavily_search` returned HTTP 200 with 3 ranked results (1 credit). Full Gemini research turn not run live |
| Weather (`weather_lookup` tool) | Open-Meteo free API: geocoding (`geocoding-api.open-meteo.com/v1/search`) + forecast (`api.open-meteo.com/v1/forecast`) | **None** (keyless for non-commercial use) | Optional (on by default) | Free for **non-commercial** use: fewer than 10,000 calls/day, 5,000/hour, 600/minute; data licensed **CC BY 4.0**, credit "Weather data by Open-Meteo.com" (terms and licence pages checked 2026-10-14). 2 calls per lookup | None (OmniFlow never uses the commercial API) | `OMNIFLOW_WEATHER_ENABLED` (on/off, not a secret) | **Verified live** 2026-10-14: one lookup ("Berlin, Germany", 2 days) returned current conditions and a 2-day forecast in Europe/Berlin time. Full Gemini weather turn not run live |
| Currency (`currency_convert` tool) | Frankfurter public API v2 (`api.frankfurter.dev/v2/rate/{FROM}/{TO}`) | **None** - no API key or account | Optional (on by default) | Free; "there are no quotas" (rate-limited only against abuse); blended reference rates free to use, including commercially (frankfurter.dev, checked 2026-10-14). 1 call per lookup | None | `OMNIFLOW_CURRENCY_ENABLED` (on/off, not a secret) | **Verified live** 2026-10-14: one conversion through `currency_convert` (1000 INR to USD) returned HTTP 200 with a rate dated 2026-10-13 in 1.5 s. Full Gemini currency turn not run live |
| Tutorial stock tool only (not OmniFlow) | Alpha Vantage | API key | Optional | Free key available | None | `ALPHAVANTAGE_API_KEY` | Not needed |
| Original tutorial scripts only (not OmniFlow) | OpenAI | API key | Optional | No free API tier | Paid usage | `OPENAI_API_KEY` | Not needed |

Phases 1-6 and the memory, coding, planner and data milestones need **only** the Gemini key; image generation optionally uses Cloudflare credentials that you create yourself (Gmail and Google Calendar were removed). Research search optionally uses a Tavily key you create yourself (added 2026-10-14); no new Python dependency was needed (it uses `httpx`, already installed). Weather (Open-Meteo, added 2026-10-14) and currency conversion (Frankfurter, added 2026-10-14) need no key, account or dependency.

**About Frankfurter (currency):** rates are mid-market *reference* rates blended from central
banks and official sources, not live trading quotes. Bank, card and remittance rates differ.
Observed behaviour (read-only requests, 2026-10-14):
- The pair endpoint returns `{"date","base","quote","rate"}`.
- An unknown currency or invalid date returns HTTP 422.
- A date with no data (future, or before coverage) returns HTTP 404, and OmniFlow then
  reports "no published rate" rather than using another day.
- A Saturday date returned a rate dated that Saturday.

**About `ddgs` (web search):** MIT licence; its README says "This library is for
educational purposes only." It's not an official DuckDuckGo API. OmniFlow pins it to the
`duckduckgo` backend, so it never contacts Google, Bing, Yandex or the other engines it can
scrape. Each search takes at most 5 results and a 10 s HTTP timeout (15 s wall clock), with one
retry on transient errors.

**About Tavily (research search):** added 2026-10-14 at your request, using a key you created.
It is read-only and sends only the search query. OmniFlow asks for ranked results only (no
generated answer, raw content or images) at `search_depth=basic`. Missing key, 401, 400/422,
429, 432/433 (plan or pay-as-you-go limit), 5xx, timeouts, network errors and malformed or
empty responses each give a specific message. Only 429, 5xx, timeouts and network errors are
retried, once. The key is read through the typed settings (`SecretStr`) and is never logged.

Gmail and Google Calendar are not integrated (removed 2026-10-13). Image generation uses
Cloudflare Workers AI because Gemini image models are **paid-only**.

**Your current model setting:** `.env` sets `GEMINI_MODEL=gemini-3.5-flash-lite` (chosen
for speed after the Phase 1 latency audit). With the fallback set to the same model, the
fallback is disabled. The code default (`gemini-3.8-flash`) is unchanged and applies only if
`GEMINI_MODEL` is unset.

## News, trends and academic research (Phase 5)

All Phase 5 providers are free and keyless. Nothing was added that needs an account,
billing or a key.

| Provider | Used for | Access (checked 2026-10-12) | Limits and caveats | Official URL |
|---|---|---|---|---|
| **OpenAlex** | Paper metadata and abstracts, all fields | Free; no key needed ("you can make basic queries with no key at all") | Keyless budget ≈ 1,000 credits/day; search = 10 credits, so about **100 searches/day**. A **free** OpenAlex key gives 10×, but it's *not* used; adding one would be a new credential needing your approval. Paid plans exist; OmniFlow never uses them | <https://help.openalex.org/how-to-use-the-api/rate-limits-and-authentication> |
| **arXiv API** | Preprints (CS, physics, maths, q-bio, …) | Free; no key | "No more than one request every three seconds", one connection, enforced in code. Metadata is CC0. Full texts are copyrighted: OmniFlow links to abstract pages and never stores or serves PDFs | <https://info.arxiv.org/help/api/tou.html> |
| **DuckDuckGo News** (`ddgs`) | Current news discovery with publication dates | No API and no account; unofficial scraper (MIT; "educational purposes only") | Can rate-limit or silently return nothing; observed doing so after repeated tests. OmniFlow reports that as "no results from DuckDuckGo News" rather than "no news exists" | <https://github.com/deedy5/ddgs> |
| **Official RSS feeds** | Institutional announcements | Public RSS published by each institution for subscribers | Fixed allowlist in code (users and the model can't add URLs). Cached 15 min, max 3 MB, 12 s timeout. Each feed has its publisher's own reuse terms; OmniFlow shows headlines, short summaries and links back, and doesn't republish articles | WHO, NASA, RBI, European Commission, arXiv RSS |

**Evaluated and not used:**
- **Semantic Scholar:** its keyless pool is shared by all users and is throttled under load, so it isn't reliable without a key.
- **Europe PMC:** its documentation page refused automated fetching, so terms couldn't be verified.
- **Crossref:** not needed, because OpenAlex already covers DOIs.
- **PIB (India):** served an HTML page instead of RSS to automated clients, so it was removed rather than worked around.
- **ISRO, UN News:** their feed URLs returned 404.

**Privacy:** search terms are sent to the provider you query (DuckDuckGo, OpenAlex, arXiv) as
ordinary web requests. They contain only your question's topic: never document contents,
data, file paths or secrets. Feeds are downloaded whole and filtered locally.

**Security:** only HTTPS URLs on a code-defined host allowlist are fetched. Redirects to other
hosts are refused, and hosts that resolve to private, loopback, link-local (including the
169.254.169.254 cloud-metadata address) or carrier-grade NAT addresses are refused. NAT64
addresses (`64:ff9b::/96`) are judged by the IPv4 address they embed. XML is parsed with
defusedxml, and every response has a size cap and timeout.

## Document embeddings (Phase 3)

| Field | Value |
|---|---|
| Purpose | Turn PDF passages and questions into vectors for retrieval (`document_search`) |
| Default model | `BAAI/bge-small-en-v1.5` (384 dimensions, 512-token input, 33.4M parameters) |
| Files actually downloaded | Hugging Face repo `Qdrant/bge-small-en-v1.5-onnx-Q`: Qdrant's quantised ONNX conversion of BAAI's model (~67 MB), via `fastembed` 0.9.0 |
| Licence | Model: MIT. `fastembed`: Apache-2.0. `onnxruntime`: MIT |
| Execution | **Local CPU**, no GPU. Document text never leaves your machine for embedding |
| Credential | None (public download, no Hugging Face token) |
| Measured cost on this PC | +228 MB RAM with the model loaded; first PDF 8.6 s (model load), then under 1 s per small PDF; about 0.18 s per query; about 14 passages/s while indexing |
| Storage | `data/models/` (gitignored). Override with `OMNIFLOW_MODEL_CACHE_DIR` |
| Privacy | Embeddings are local. When you ask a question, only the top retrieved passages (not whole files) are sent to Gemini to write the answer, and Gemini's free-tier data terms apply to those passages |
| Billing risk | None |
| Official URLs | <https://huggingface.co/BAAI/bge-small-en-v1.5> · <https://qdrant.github.io/fastembed/> |

**Why this model:** its model card reports MTEB retrieval 51.68, and it handles 512 tokens,
so roughly 1,000-character chunks aren't cut off. It's MIT-licensed and needs no PyTorch.

**Alternatives considered:**
- `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 90 MB) is allowed as a setting. Its
  card says input is cut off at 256 word pieces (it was trained at 128).
- Running MiniLM through `sentence-transformers` was rejected: it would install PyTorch,
  transformers, scipy and scikit-learn (roughly 1-2 GB).
- Gemini embeddings were rejected for documents: they're remote, and free-tier content may be
  used by Google and read by human reviewers.

Only the two local models above are accepted by the config. No remote embedding provider
is configured.

**Relevance threshold:** `OMNIFLOW_RAG_MIN_SCORE=0.55`, calibrated on this model. Relevant
passages scored 0.71-0.85 and unrelated questions 0.38-0.47. bge scores cluster high, as its
model card notes, so re-tune this if you switch models.

## Data analysis and SQL (Phase 4): local libraries only

| Library | Version | Licence | Role | Runs |
|---|---|---|---|---|
| pandas / NumPy | 3.0.6 / 2.5.3 | BSD-3 | every calculation (already installed with Streamlit) | local |
| Altair | 6.3.0 | BSD-3 | chart specs, drawn in the browser by Streamlit (already installed) | local |
| openpyxl | 3.1.5 | MIT | reads `.xlsx` files; formulas are read as stored values, never evaluated | local |
| defusedxml | 0.7.1 | PSF-2.0 | used automatically by openpyxl to block malicious XML (e.g. entity bombs) | local |
| sqlglot | 30.22.0 | MIT | parses SQL into a syntax tree so queries can be validated before running | local |
| sqlite3 | stdlib (SQLite 3.42.0) | public domain | the database engine; no server | local |

Matplotlib was not added (Altair was already installed). No model was downloaded for
Phase 4, and none of these libraries contacts the network.

**What Gemini does here:** it maps your question onto one allowlisted operation (or writes a
SQL SELECT that is then validated), and it explains the computed results. It never computes
numbers: those come from pandas or SQLite, and the verifier flags any number in an answer
that isn't in the computed output. A data question typically uses 2 free-tier requests.
The quick-chart builder, profile, preview and "Run your own SELECT" make **no** Gemini call.

**What is sent to Gemini:** for data questions, dataset and column names with their types,
plus the computed result table (at most 30 rows), never the whole file. For SQL, the
allowlisted table and column names and the query result rows.

**Your imported data (SQL):** "Import CSV/XLSX into SQL" copies a dataset into a private
SQLite file for your browser session, in the operating system's temp folder (outside the
project). It uses the same local libraries, so it adds no service and no credential. The
file is deleted when you delete the imports, after 120 minutes idle, or when the app stops.

**Sample database:** `data/sample/omniflow_sample.db` (gitignored) is generated locally
with fixed, fictional data. It's labelled "SAMPLE DATA" everywhere, and it contains a
non-allowlisted `staff_payroll` table and a restricted `customers.email` column to
demonstrate the access controls.

## OCR (optional, Phase 3)

Tesseract is a free, local OCR program (Apache-2.0). OmniFlow uses it only for PDF pages
that contain images but no text, at most `OMNIFLOW_OCR_MAX_PAGES` (20) pages per file, with
a 30 s timeout per page. It's called as a local program with a fixed argument list (no
shell, no network). To enable it, install it yourself and restart the app:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

It's detected on `PATH` or in `C:\Program Files\Tesseract-OCR\`, or you can set
`OMNIFLOW_TESSERACT_CMD`. **Status: not installed on this machine, so real OCR is untested.**
Without it, text PDFs work normally and scanned pages are reported as not searchable.

## Gemini: model selection

Source: official pages checked 2026-10-09:
[pricing](https://ai.google.dev/gemini-api/docs/pricing) (updated 2026-10-09),
[models](https://ai.google.dev/gemini-api/docs/models) (updated 2026-10-06),
[gemini-3.8-flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash),
[terms](https://ai.google.dev/gemini-api/terms).

| Model | Role | Why |
|---|---|---|
| `gemini-3.8-flash` | **Default** (`GEMINI_MODEL`) | Newest **stable** model with a free tier. 1,048,576-token input, 65,536 output, function calling, structured outputs, thinking, text/image/video/audio/PDF input, all needed in later phases |
| `gemini-3.5-flash-lite` | **Fallback** (`GEMINI_FALLBACK_MODEL`) | Stable and free, with the same context and capability list. Lighter, so it's useful when the primary is rate-limited. Used only on rate-limit, model-unavailable or server errors, never on auth errors |
| `gemini-3.1-pro-preview` | Not used | Strongest reasoning, but **paid-only** (no free tier) |
| `gemini-2.5-*` | Not used | Access is limited to past users per the models page |

Both models can be changed in `.env` without code changes. The app shows which model
answered each message.

## Gemini: free-tier conditions you should know

- **Data use:** on the unpaid tier Google may use prompts and responses, **including images
  you ask about**, to improve its products, and human reviewers may read them. Google's terms say: *do not submit sensitive,
  confidential, or personal information to the Unpaid Services.*
- **Region:** apps made available to users in the EEA, Switzerland or the UK may use only
  Paid Services.
- **Age:** API users must be 18+.
- **Limits:** free-tier RPM/TPM/RPD limits are shown only in Google AI Studio for your project.
  OmniFlow caps cost with `GEMINI_MAX_OUTPUT_TOKENS` (8192), `GEMINI_TIMEOUT_SECONDS` (60),
  `GEMINI_MAX_RETRIES` (2), `OMNIFLOW_MAX_HISTORY_MESSAGES` (40) and
  `OMNIFLOW_MAX_INPUT_CHARS` (8000).
- Free-tier terms can change, so re-check the pricing page before relying on them.

## How to configure the Gemini key (Windows)

1. Open <https://aistudio.google.com/apikey> and sign in with your Google account.
2. Accept the terms. AI Studio creates a default project and key, or click **Create API key**.
   Do **not** enable billing.
3. In the project folder, create your local env file (it is gitignored):
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
4. Put the key after `GEMINI_API_KEY=` (no quotes, no spaces), then save.
   Do not paste the key into chat, code, or any other file.
5. Verify it (uses one request per configured model; never prints the key):
   ```powershell
   .\.venv\Scripts\python.exe -m omniflow.diagnostics --live
   ```
6. Start or restart the app. Settings are read once at startup.

If the key ever leaks, delete it in AI Studio, create a new one, and update `.env`.

## Image understanding (Phase 6)

- **Provider:** Google Gemini, through the existing `GEMINI_API_KEY` and the model in
  `GEMINI_MODEL` (yours: `gemini-3.5-flash-lite`). No new key, account or billing.
- **Processing:** remote, on Google's servers. Only a cleaned copy is sent: re-encoded,
  at most 1536 px, with metadata removed.
- **Cost:** free on the free tier (verified on the pricing page and with one live request,
  2026-10-13). Each image question uses one extra request from the same free quota as chat.
- **Why not a local model:** this PC has no CUDA GPU and about 1-2 GB of free memory.
  PyTorch plus a small vision model would need several GB, be slow on the CPU and read
  text and charts noticeably worse. A local option can be proposed later if you want
  offline processing.
- **Formats:** JPEG, PNG, WebP. Gemini also accepts HEIC/HEIF, but OmniFlow doesn't, because
  Pillow can't decode them without an extra plugin.

## Hosting and durable storage (memory milestone)

- **Streamlit Community Cloud:** a free hosting service (Streamlit's documentation, checked
  2026-10-13). Secrets are entered in the app settings, and root-level secrets become
  environment variables, so `GEMINI_API_KEY` works unchanged.
- **Documented limits** (as of February 2024, "may change at any time"):
  - up to 2 CPU cores, 2.7 GB memory and 50 GB storage
  - apps sleep after 12 hours without traffic
- **Local files aren't documented as persistent.** OmniFlow treats the disk there as
  temporary.
- **Nothing was provisioned** and no hosted database is used.

**Options for durable hosted memory.** None is implemented, none has been checked for free
status, and none will be added without your approval:

| Option | Needs | Notes |
|---|---|---|
| Keep using OmniFlow locally (current) | Nothing | Durable on your PC; this is the only fully working setup today |
| Export / import memories as a file | No service (code change only) | You download your memories and upload them on the next visit. Works on Community Cloud without any account |
| A hosted database with a free tier (e.g. a managed Postgres or libSQL service) | An account, a connection credential, a storage adapter, **and per-user sign-in** | Free-tier limits and terms must be verified first. Without sign-in, a shared app cannot tell users apart |
| Streamlit's built-in user authentication (OIDC) | An identity-provider client (credential) | Gives a stable user id, but on its own still doesn't make the disk durable |

## Image generation (Cloudflare Workers AI)

Only image generation uses this provider. Gmail and Google Calendar integrations were
removed on 2026-10-13, so no Google OAuth client is needed.

**Steps:**
1. Create a free account: https://dash.cloudflare.com/sign-up
2. Copy your 32-character Account ID.
3. Create an **API token** (not the Global API Key) with the **Workers AI** permission for
   that account: https://dash.cloudflare.com/profile/api-tokens
4. Set `OMNIFLOW_IMAGE_PROVIDER=cloudflare`, `CLOUDFLARE_ACCOUNT_ID` and
   `CLOUDFLARE_API_TOKEN` in `.env`, then restart.

**Free plan:** 10,000 Neurons a day. Prompts are sent to Cloudflare. How Cloudflare retains
prompts isn't verified here; check Cloudflare's terms.

**What an HTTP 401 means:** Cloudflare didn't accept the token. Typical causes:
- the Global API Key was used instead of an API token;
- the token was copied incompletely or with extra spaces or quotes;
- the token was rolled or deleted, or isn't active yet.

An HTTP 403 means the token is valid but lacks Workers AI permission for that account.

Never paste these values into chat or code. On a hosted deployment, use the platform's
secrets manager.
