# Multi-Provider OAuth Bridge

Bridge OpenAI-compatible chạy local, hỗ trợ **4 provider**:

- **ChatGPT / Codex** (account ChatGPT) → endpoint `chatgpt.com/backend-api/codex/responses`
- **Google Gemini** (account Google AI Pro) → endpoint `cloudcode-pa.googleapis.com` (Code Assist)
- **Anthropic Claude** (API key Anthropic Console) → endpoint `api.anthropic.com/v1/messages`
- **DeepSeek** (API key DeepSeek) → endpoint `api.deepseek.com/chat/completions`

Mọi client OpenAI-compatible (LangChain, LibreChat, OpenWebUI, Cherry Studio, openai SDK, ...) chỉ cần trỏ vào `http://127.0.0.1:12345/v1`.

## Tính năng nổi bật

- **Multi-account pool**: thêm nhiều account cùng loại (vd 3 account ChatGPT) hoặc khác loại (ChatGPT + Gemini + Claude + DeepSeek) trong 1 bridge.
- **Auto failover khi 429 / 401**: account hết quota hoặc token hỏng → tự chuyển account/model khác khi route đang bật quay vòng, **transparent** với client.
- **Auto-route theo model**: client gửi `gpt-5.5` → ChatGPT pool, `gemini-2.5-pro` → Google pool, `claude-sonnet-4-6` → Anthropic pool, `deepseek-v4-flash` → DeepSeek pool. **Không cần restart** khi đổi model.
- **Route groups**: tạo model alias tuỳ biến từ provider, model cụ thể, hoặc nhóm khác; chọn thứ tự ưu tiên / round-robin / random. Model đầu tiên được thử trước ở chế độ ưu tiên.
- **Cross-provider fallback** (tuỳ chọn): khi pool chính cạn quota, tự nhảy sang provider khác.
- **Sức mạnh tối đa từng provider**:
  - Codex: reasoning summary stream, structured output (JSON schema strict), tool calling honor strict, built-in tools (`web_search`, `file_search`, `code_interpreter`), `previous_response_id`, multimodal PDF/image/audio/video, usage thật (gồm `reasoning_tokens` + `cached_tokens`).
  - Gemini: thinking tokens, function calling, response JSON schema.
  - Claude: extended thinking (`thinking_delta`), tool_use streaming, cache_read tokens, PDF documents.
- **Performance**: shared `httpx.AsyncClient` (HTTP/2), async lock, SSE keepalive.
- **UI quản lý nhiều slot** với badge sức khoẻ realtime: healthy / 429 / expired / disabled.
- **Rate/quota visibility**: đọc header rate-limit/quota của GPT, Gemini, Claude, DeepSeek và hiển thị ở UI + `/api/accounts`.

## Cài & chạy

```bash
pip install -r requirements.txt
python main.py
```

Mở `http://localhost:12345` → bấm **Add account** (chọn provider + alias) → bấm **Login** ở slot vừa tạo → đăng nhập trình duyệt.

Launcher có sẵn: `run.bat` (Windows), `start.bat`/`stop.bat` (Windows hidden), `bash run.sh` (Linux/macOS).

## Kiến trúc

```
standalone/
├── main.py              # Entry point (uvicorn)
├── bridge.py            # FastAPI app: routes, failover loop, UI
├── core.py              # Config, shared HTTP client, SSE, errors, OAuth helpers
├── accounts.py          # Account, AccountPool, routing, migration
├── providers/
│   ├── codex.py         # ChatGPT/Codex: OAuth + Responses API mapping
│   ├── google.py        # Google Code Assist: OAuth + Gemini mapping
│   ├── anthropic.py     # Anthropic: API key + Messages API mapping
│   └── deepseek.py      # DeepSeek: API key + OpenAI-compatible Chat Completions
├── data/
│   ├── accounts/
│   │   ├── chatgpt-1/   # mỗi slot 1 folder
│   │   │   ├── oauth.json
│   │   │   └── meta.json
│   │   ├── chatgpt-2/
│   │   ├── google-1/
│   │   └── claude-1/
│   ├── pool_state.json  # health state (exhausted_until per slot)
│   ├── bridge.log
│   └── bridge.pid
└── requirements.txt
```

Khi upgrade từ bản cũ: `data/oauth.json` / `data/google_oauth.json` được **tự migrate** sang `data/accounts/{provider}-default/` ở lần khởi động đầu tiên — không mất session.

## Endpoints

### OpenAI-compatible

| Method | Path | Mục đích |
|---|---|---|
| GET | `/v1/models` | Danh sách model từ tất cả pool đang có |
| POST | `/v1/chat/completions` | Chat (stream + non-stream) với failover |
| POST | `/v1/audio/transcriptions` | Whisper-style → Codex audio backend |
| GET/POST | `/v1/oauth/token` | Bearer token sống (?slot_id=... hoặc ?provider=...) |
| GET | `/.well-known/openai-bridge` | Discovery |

### Quản lý account

| Method | Path | Mục đích |
|---|---|---|
| GET | `/api/accounts` | List slot + health |
| POST | `/api/accounts` | `{provider, alias}` → tạo slot mới |
| DELETE | `/api/accounts/{slot_id}` | Xoá slot |
| PATCH | `/api/accounts/{slot_id}` | `{alias?, enabled?, tier?}` |
| POST | `/api/accounts/{slot_id}/login` | Mở OAuth flow cho ChatGPT/Gemini |
| POST | `/api/accounts/{slot_id}/api-key` | Lưu API key cho slot Claude/DeepSeek |
| POST | `/api/accounts/{slot_id}/refresh` | Force refresh token |
| POST | `/api/accounts/{slot_id}/logout` | Clear token (giữ slot) |
| GET | `/api/groups` | List route group + model/provider tag |
| POST | `/api/groups` | Lưu route group `{name, mode, items}` |
| DELETE | `/api/groups/{name}` | Xoá route group |
| GET | `/` | UI |
| GET | `/health` | Liveness probe |

## Routing theo model

Bridge tự suy provider từ tên model (prefix-based):

| Prefix model | Provider | Default model |
|---|---|---|
| `gpt-*`, `o1`, `o3`, `o4`, `codex-*`, `chatgpt-*`, `5.*` | `chatgpt` | `gpt-5.5` |
| `gemini-*`, `auto-gemini-*`, `models/gemini-*` | `google` | `auto-gemini-3` |
| `claude-*` | `anthropic` | `claude-sonnet-4-6` |
| `deepseek-*` | `deepseek` | `deepseek-v4-flash` |

Aliases được chấp nhận: xem `providers/*.py` (vd `gpt-5` → `gpt-5.5`, `claude-opus-latest` → `claude-opus-4-7`, ...).

Bạn cũng có thể dùng chính tên provider như model (`chatgpt`, `google`, `anthropic`, `deepseek`) để quay vòng trong provider đó, hoặc dùng tên route group tự tạo. Group có thể chứa provider, model cụ thể thuộc provider bất kỳ, hoặc group khác. Chế độ `priority` thử theo thứ tự kéo thả; `round_robin` và `random` đổi thứ tự item ở mỗi request. Khi item lỗi model, hết quota, auth lỗi, hoặc 5xx, bridge bỏ qua item đó và thử item tiếp theo.

## Failover khi 429 / 401

Khi 1 request thất bại:

| Lỗi | Hành động |
|---|---|
| 429 `usage_limit_reached` | Đánh dấu account `exhausted_until = resets_at` (parse từ response). Tự retry account khác cùng provider. |
| 429 generic rate limit | Mark exhausted ~60s. Retry account khác. |
| 401 / refresh token hỏng | Mark account `invalid`. Cần login lại. Retry account khác. |
| 5xx | Release acc, retry account khác (lỗi thường tạm thời). |
| 4xx khác (400, 403) | Với route group thì thử item tiếp theo; ngoài group thì trả lỗi cho client. |

Tối đa `BRIDGE_MAX_FAILOVER=3` account/request. Nếu mọi account đều exhausted, response sẽ chứa message giải thích kèm thời gian reset gần nhất.

## Multi-account: cách thêm 2+ account

### Cùng loại (vd 3 account ChatGPT)

1. UI → `Add account` → chọn `ChatGPT (Codex)` → alias `Personal`.
2. Click `Login` ở slot mới → đăng nhập account ChatGPT thứ nhất.
3. Lặp lại với alias `Work` (đăng nhập ở browser **incognito** hoặc browser khác để tránh dùng nhầm session).
4. Thêm nữa nếu cần.

Bridge tự load balance qua chiến lược `BRIDGE_POOL_STRATEGY` (`least_load` mặc định).

### Khác loại

Cứ làm tương tự với provider khác. UI hiển thị tất cả slot trong cùng grid.

## Biến môi trường

### Chung

| Biến | Default | Mục đích |
|---|---|---|
| `BRIDGE_HOST` | `127.0.0.1` | |
| `BRIDGE_PORT` | `12345` | |
| `BRIDGE_API_KEY` | (rỗng) | Nếu set, mọi request `/v1/*` cần header `Authorization: Bearer <key>` |
| `BRIDGE_POOL_STRATEGY` | `least_load` | `least_load` / `round_robin` / `random` |
| `BRIDGE_MAX_FAILOVER` | `3` | Max account/request |
| `BRIDGE_LOCALE` | `vi` | `vi` / `en` (error messages) |
| `BRIDGE_CROSS_PROVIDER_FALLBACK` | (rỗng) | Vd `claude,gemini` — khi provider chính cạn quota, thử provider này |
| `BRIDGE_SSE_KEEPALIVE` | `15` | Giây emit comment-line giữ stream |
| `BRIDGE_ENABLE_CORS` | `1` | CORS cho web client |
| `BRIDGE_LOG_MAX_BYTES` | `10485760` | Log rotation |

### Codex (ChatGPT)

| Biến | Default | Mục đích |
|---|---|---|
| `OPENAI_CODEX_DEFAULT_MODEL` | `gpt-5.5` | |
| `OPENAI_CODEX_DEFAULT_INSTRUCTIONS` | `You are ChatGPT, ...` | |
| `OPENAI_CODEX_REASONING_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` |
| `OPENAI_CODEX_VERBOSITY` | `medium` | `low` / `medium` / `high` |
| `OPENAI_CODEX_REASONING_SUMMARY` | (rỗng) | `auto` / `concise` / `detailed` |
| `OPENAI_CODEX_PASSTHROUGH_SAMPLING` | `0` | Forward `temperature` / `top_p` / `seed` |
| `OPENAI_CODEX_DEFAULT_STORE` | `0` | Mặc định bật `store: true` |
| `OPENAI_CODEX_DEFAULT_INCLUDE` | (rỗng) | Vd `reasoning.encrypted_content,file_search_call.results` |
| `OPENAI_CODEX_AUDIO_MODEL` | `gpt-5.4` | Model cho `/v1/audio/transcriptions` |
| `OPENAI_CODEX_VIDEO_FRAMES` | `6` | |
| `OPENAI_CODEX_VIDEO_FPS` | `1/3` | |
| `OPENAI_CODEX_VIDEO_MAX_WIDTH` | `1280` | |
| `OPENAI_CODEX_ORIGINATOR` | `codex_cli_rs` | Header `originator` |
| `OPENAI_CODEX_BETA` | (rỗng) | Header `OpenAI-Beta` |
| `OPENAI_CODEX_USER_AGENT` | (rỗng) | Override |

### Google (Gemini)

| Biến | Default | Mục đích |
|---|---|---|
| `GOOGLE_OAUTH_CLIENT_ID` | (rỗng) | OAuth client id cho Google login |
| `GOOGLE_OAUTH_CLIENT_SECRET` | (rỗng) | OAuth client secret cho Google login |
| `GOOGLE_GEMINI_DEFAULT_MODEL` | `auto-gemini-3` | |
| `GOOGLE_CODE_ASSIST_PROJECT` | (rỗng) | GCP project id |
| `GOOGLE_CODE_ASSIST_IGNORE_SERVER_PROJECT` | `0` | |
| `GOOGLE_CODE_ASSIST_SKIP_LOAD` | `0` | Skip preflight `loadCodeAssist` |
| `GOOGLE_CODE_ASSIST_USER_PROJECT_HEADER` | `0` | Gửi `x-goog-user-project` |
| `GOOGLE_OAUTH_PROMPT` | `consent` | |
| `GOOGLE_GEMINI_USER_AGENT` | `google-gemini-cli` | |

### Anthropic (Claude)

| Biến | Default | Mục đích |
|---|---|---|
| `ANTHROPIC_DEFAULT_MODEL` | `claude-sonnet-4-6` | |
| `ANTHROPIC_API_KEY` | (rỗng) | API key chính thức từ Anthropic Console; có thể nhập per-slot trên UI |
| `ANTHROPIC_BETA` | (rỗng) | Header `anthropic-beta` khi dùng API key, nếu cần |
| `ANTHROPIC_ALLOW_LEGACY_OAUTH` | `0` | Bật lại Claude Code OAuth legacy nếu tự chấp nhận rủi ro |
| `ANTHROPIC_OAUTH_BETA` | `oauth-2025-04-20` | Header beta cho legacy OAuth |
| `ANTHROPIC_CLIENT_ID` | (Claude Code public client) | Chỉ dùng khi bật legacy OAuth |
| `ANTHROPIC_AUTH_URL` | `https://platform.claude.com/oauth/authorize` | Chỉ dùng khi bật legacy OAuth |
| `ANTHROPIC_TOKEN_URL` | `https://platform.claude.com/v1/oauth/token` | Chỉ dùng khi bật legacy OAuth |

Claude hiện dùng API key chính thức. Tạo slot Claude rồi bấm **API key** trên UI, hoặc set `ANTHROPIC_API_KEY` trước khi khởi động service.

### DeepSeek

| Biến | Default | Mục đích |
|---|---|---|
| `DEEPSEEK_DEFAULT_MODEL` | `deepseek-v4-flash` | |
| `DEEPSEEK_API_KEY` | (rỗng) | API key DeepSeek; có thể nhập per-slot trên UI |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Override nếu cần proxy |

DeepSeek dùng OpenAI-compatible API. Model hiện dùng: `deepseek-v4-flash`, `deepseek-v4-pro`; `deepseek-chat` và `deepseek-reasoner` là alias legacy dự kiến bị bỏ sau **2026-07-24** theo docs DeepSeek.

Rate/quota được đọc từ response headers của provider và lưu vào `health.rate_limit` trong `/api/accounts`; UI hiển thị riêng ở từng slot, không đưa thông tin này vào nội dung assistant.

## Ví dụ dùng

### Cùng client, đổi model giữa các provider

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:12345/v1", api_key="not-needed")

# Gọi ChatGPT
client.chat.completions.create(model="gpt-5.5", messages=[{"role": "user", "content": "hi"}])

# Gọi Gemini
client.chat.completions.create(model="gemini-2.5-pro", messages=[{"role": "user", "content": "hi"}])

# Gọi Claude
client.chat.completions.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}])

# Gọi DeepSeek
client.chat.completions.create(model="deepseek-v4-flash", messages=[{"role": "user", "content": "hi"}])

# Gọi route group tự tạo trên UI, ví dụ group tên "all"
client.chat.completions.create(model="all", messages=[{"role": "user", "content": "hi"}])
```

Bridge tự pick account và route, client không cần biết provider.

### Cross-provider fallback

```bash
BRIDGE_CROSS_PROVIDER_FALLBACK=claude,gemini python main.py
```

Khi user gọi `gpt-5.5` và tất cả account ChatGPT đều `usage_limit_reached`, bridge tự thử Claude → Gemini.

### Bridge API key auth

```bash
BRIDGE_API_KEY=mysecret python main.py
```

Client phải gửi `Authorization: Bearer mysecret` ở mọi request `/v1/*`.

## Troubleshooting

| Triệu chứng | Cách xử lý |
|---|---|
| `No account for provider chatgpt` | UI → Add account → Login |
| Account `invalid` | UI → Login lại |
| Tất cả 429 | Đợi reset (UI hiển thị `429 23s`) hoặc thêm account khác |
| Claude chưa chạy | UI → slot Claude → API key, hoặc set `ANTHROPIC_API_KEY` |
| Stream cắt giữa | Tự retry 502/503/504. Tăng `BRIDGE_UPSTREAM_RETRIES` nếu cần |

Log: `data/bridge.log` (auto-rotate). Tail trên Windows: `Get-Content data\bridge.log -Wait -Tail 50`.
