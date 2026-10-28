from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Remote Lab")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Remote Lab</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 20px;
            background: #f1f5f9;
            font-family: Arial, sans-serif;
            color: #1e293b;
        }
        .container {
            max-width: 480px;
            margin: 25px auto;
        }
        h1 { font-size: 26px; margin-bottom: 5px; }
        .subtitle { color: #64748b; }
        .card {
            background: white;
            border-radius: 14px;
            padding: 22px;
            margin-top: 16px;
            box-shadow: 0 2px 8px #0000000d;
        }
        .online { color: #16a34a; font-weight: bold; }
        .offline { color: #dc2626; font-weight: bold; }
        .muted { color: #64748b; }
        button {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        .note { font-size: 13px; color: #64748b; }
    </style>
</head>
<body>
    <main class="container">
        <h1>Remote Lab</h1>
        <p class="subtitle">모바일 테스트 관리 시스템</p>

        <div class="card">
            <h3>Render 서버</h3>
            <p>연결 상태: <span id="server-status">확인 중</span></p>
            <p class="note" id="server-time"></p>
        </div>

        <div class="card">
            <h3>테스트 PC</h3>
            <p>연결 상태: <span class="muted">미연결</span></p>
            <p class="note">테스트 PC 연결 기능은 아직 구현되지 않았습니다.</p>
        </div>

        <div class="card">
            <button onclick="checkServer()">서버 상태 확인</button>
        </div>
    </main>

    <script>
        async function checkServer() {
            const status = document.getElementById("server-status");
            const time = document.getElementById("server-time");
            try {
                const response = await fetch("/health", {cache: "no-store"});
                if (!response.ok) throw new Error("HTTP error");
                const data = await response.json();
                status.textContent = "정상 작동";
                status.className = "online";
                time.textContent = "서버 시간(UTC): " + data.timestamp;
            } catch (error) {
                status.textContent = "연결 실패";
                status.className = "offline";
                time.textContent = "서버 상태를 확인할 수 없습니다.";
            }
        }
        checkServer();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE

@app.get("/health")
async def health():
    return {"status": "online", "timestamp": datetime.now(timezone.utc).isoformat()}
