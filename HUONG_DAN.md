# Hướng dẫn sử dụng — Multi-Provider OAuth Bridge

Bridge chạy độc lập trên port **12345**, hỗ trợ đăng nhập/lưu API key **nhiều account** cho **4 provider** (ChatGPT, Google Gemini, Anthropic Claude, DeepSeek) cùng lúc, **tự failover** khi 1 account/model hết quota hoặc lỗi.

---

## 1. Yêu cầu

- Python **3.10+**
- 1 hoặc nhiều account ở các provider:
  - ChatGPT Plus / Pro / Team (account miễn phí KHÔNG dùng được Codex backend)
  - Google AI Pro (cho Gemini)
  - Anthropic Console API key (cho Claude)
  - DeepSeek API key (cho DeepSeek)
- Trình duyệt mặc định để đăng nhập OAuth cho ChatGPT/Gemini

---

## 2. Cài đặt & chạy

```bat
cd standalone
run.bat            :: Windows foreground
start.bat          :: Windows hidden (stop.bat để dừng)
```

```bash
bash run.sh        # Linux / macOS
```

Lần đầu tự cài `pip install -r requirements.txt`. Sau đó bridge chạy ở `http://localhost:12345`.

---

## 3. Thêm account

1. Mở `http://localhost:12345`
2. Chọn provider (ChatGPT / Google / Anthropic / DeepSeek), gõ alias (vd `Personal`, `Work`) → **Create slot**
3. ChatGPT/Gemini: bấm **Login** ở slot vừa tạo → tab mới mở OAuth → đăng nhập → tab tự đóng
4. Claude/DeepSeek: bấm **API key** ở slot vừa tạo rồi dán API key tương ứng
5. Lặp lại để thêm account khác. **Mẹo**: dùng incognito hoặc browser khác để đăng nhập account ChatGPT thứ hai mà không bị conflict session.

Bridge tự load balance khi có nhiều slot cùng provider, tự failover khi 1 slot bị 429. Slot chỉ cần bật **Enabled** là có thể chạy; **quay vòng** chỉ là tuỳ chọn ưu tiên chọn tải khi dùng pool nhiều account.

### Nhóm route tuỳ biến

Trong UI phần **Nhóm route**, kéo hoặc bấm tag để tạo nhóm:

- Tag provider: `chatgpt`, `google`, `anthropic`, `deepseek` để quay vòng trong provider đó.
- Tag model: chọn model cụ thể như `gpt-5.5`, `gemini-2.5-pro`, `claude-sonnet-4-6`, `deepseek-v4-pro`.
- Tag group: kéo cả nhóm đã tạo vào nhóm khác.

Chọn mode:

- **Ưu tiên theo thứ tự**: item đầu tiên chạy trước, lỗi/quota thì bỏ qua item kế.
- **Quay vòng**: mỗi request bắt đầu từ item kế tiếp.
- **Ngẫu nhiên**: mỗi request xáo thứ tự item.

Sau khi lưu nhóm, dùng tên nhóm làm `model`, ví dụ `model="all"`.

---

## 4. Dùng từ client

Bất kỳ client OpenAI-compatible nào, base URL:

```
http://127.0.0.1:12345/v1
```

API key: gì cũng được (`not-needed`) — trừ khi bạn set `BRIDGE_API_KEY` thì cần đúng key đó.

### Đổi model giữa các provider

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:12345/v1", api_key="x")

c.chat.completions.create(model="gpt-5.5", ...)             # → ChatGPT pool
c.chat.completions.create(model="gemini-2.5-pro", ...)      # → Google pool
c.chat.completions.create(model="claude-sonnet-4-6", ...)   # → Anthropic pool
c.chat.completions.create(model="deepseek-v4-flash", ...)   # → DeepSeek pool
c.chat.completions.create(model="all", ...)                 # → route group tên all
```

Bridge tự suy provider từ tên model, không cần restart.

---

## 5. Khi gặp lỗi

| Triệu chứng | Cách xử lý |
|---|---|
| Một slot `429 23s` | Đợi 23s reset hoặc thêm slot khác cùng provider |
| Một slot `invalid` (token revoked) | Bấm **Login** lại |
| Tất cả slot hết quota | Bật cross-provider fallback: `BRIDGE_CROSS_PROVIDER_FALLBACK=claude,gemini` |
| Claude chưa chạy | Tạo slot Claude rồi bấm **API key**, hoặc set `ANTHROPIC_API_KEY` trong môi trường service |
| DeepSeek chưa chạy | Tạo slot DeepSeek rồi bấm **API key**, hoặc set `DEEPSEEK_API_KEY` trong môi trường service |
| Stream cắt giữa chừng | Bridge tự retry 502/503/504; tăng `BRIDGE_UPSTREAM_RETRIES` nếu cần |

Quota/rate của GPT, Gemini, Claude, DeepSeek nằm ở dòng `rate/quota` trên UI và trong `health.rate_limit` của `/api/accounts`; bridge không chèn thông tin này vào nội dung trả lời.

Log realtime:
```powershell
Get-Content data\bridge.log -Wait -Tail 50
```

---

## 6. Migration từ bản cũ

Nếu bạn upgrade từ bản single-account cũ:
- `data/oauth.json` → tự chuyển sang `data/accounts/chatgpt-default/oauth.json`
- `data/google_oauth.json` → tự chuyển sang `data/accounts/google-default/oauth.json`
- Session cũ giữ nguyên, không cần login lại

Mọi env var cũ (`OPENAI_CODEX_*`, `GOOGLE_*`) đều còn dùng được. Xem `README.md` cho danh sách đầy đủ.

Với Google login, đặt `GOOGLE_OAUTH_CLIENT_ID` và `GOOGLE_OAUTH_CLIENT_SECRET` trong môi trường trước khi chạy bridge.
