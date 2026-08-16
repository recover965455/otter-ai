//! Minimal local HTTP mock server used by the Codex e2e tests.
//!
//! Hand-rolled on `tokio::TcpListener` so the test suite needs no extra
//! dev-dependencies. Supports:
//!
//! * request capture (method / path / headers / body)
//! * full canned responses (status + headers + body)
//! * streaming chunked responses with per-chunk delays and optional
//!   keep-open (to emulate SSE bodies that never close)
//! * connections that never answer (for response-header timeout tests)

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, Clone)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    /// Decompressed request body (zstd bodies are transparently decoded).
    pub body: Vec<u8>,
    /// Raw bytes as read from the wire (kept for compression assertions).
    pub raw_body: Vec<u8>,
}

impl RecordedRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(|s| s.as_str())
    }

    pub fn json_body(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).expect("request body is valid JSON")
    }

    pub fn form_body(&self) -> HashMap<String, String> {
        let text = String::from_utf8_lossy(&self.body).to_string();
        let mut out = HashMap::new();
        for pair in text.split('&') {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            out.insert(url_decode(k), url_decode(v));
        }
        out
    }
}

fn maybe_decompress_request_body(headers: &HashMap<String, String>, body: Vec<u8>) -> Vec<u8> {
    #[cfg(feature = "codex-zstd")]
    {
        let encoding = headers
            .get("content-encoding")
            .map(|s| s.as_str())
            .unwrap_or("");
        if encoding.contains("zstd") {
            return zstd::decode_all(std::io::Cursor::new(body)).unwrap_or_default();
        }
    }
    body
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

#[derive(Debug, Clone)]
pub struct SseChunk {
    pub data: String,
    pub delay: Duration,
}

impl SseChunk {
    pub fn now(data: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            delay: Duration::from_millis(0),
        }
    }

    pub fn after(data: impl Into<String>, delay: Duration) -> Self {
        Self {
            data: data.into(),
            delay,
        }
    }
}

#[derive(Clone)]
pub enum MockResponse {
    /// A complete response with status / headers / body (Content-Length).
    Full {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    },
    /// A chunked streaming response; `keep_open` never terminates the body.
    Stream {
        status: u16,
        headers: Vec<(String, String)>,
        chunks: Vec<SseChunk>,
        keep_open: bool,
    },
    /// Accepted but never answered (headers never arrive).
    Silent,
}

impl MockResponse {
    pub fn json(status: u16, value: serde_json::Value) -> Self {
        MockResponse::Full {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: serde_json::to_vec(&value).unwrap(),
        }
    }

    pub fn json_with_headers(
        status: u16,
        value: serde_json::Value,
        headers: Vec<(String, String)>,
    ) -> Self {
        MockResponse::Full {
            status,
            headers,
            body: serde_json::to_vec(&value).unwrap(),
        }
    }

    pub fn text(status: u16, body: impl Into<String>) -> Self {
        MockResponse::Full {
            status,
            headers: vec![("content-type".into(), "text/plain".into())],
            body: body.into().into_bytes(),
        }
    }

    /// Join pre-formatted SSE lines (each already starting with `data:` …)
    /// into a complete `text/event-stream` body.
    pub fn sse(events: Vec<String>) -> Self {
        let body = events
            .iter()
            .map(|e| format!("{}\n\n", e))
            .collect::<Vec<_>>()
            .join("");
        MockResponse::Full {
            status: 200,
            headers: vec![
                ("content-type".into(), "text/event-stream".into()),
                ("cache-control".into(), "no-cache".into()),
            ],
            body: body.into_bytes(),
        }
    }

    pub fn sse_stream(chunks: Vec<SseChunk>, keep_open: bool) -> Self {
        MockResponse::Stream {
            status: 200,
            headers: vec![
                ("content-type".into(), "text/event-stream".into()),
                ("cache-control".into(), "no-cache".into()),
            ],
            chunks,
            keep_open,
        }
    }
}

pub type RequestHandler = Arc<dyn Fn(&RecordedRequest) -> MockResponse + Send + Sync>;

pub struct MockServer {
    pub addr: std::net::SocketAddr,
    pub requests: Arc<Mutex<Vec<RecordedRequest>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl MockServer {
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn recorded(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    pub async fn spawn(handler: RequestHandler) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let addr = listener.local_addr().expect("local addr");
        let requests: Arc<Mutex<Vec<RecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let reqs = requests.clone();

        let handle = tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let handler = handler.clone();
                let reqs = reqs.clone();
                tokio::spawn(async move {
                    let _ = handle_connection(stream, handler, reqs).await;
                });
            }
        });

        Self {
            addr,
            requests,
            handle,
        }
    }

    pub fn shutdown(self) {
        self.handle.abort();
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    handler: RequestHandler,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
) -> std::io::Result<()> {
    loop {
        let request = match read_request(&mut stream).await? {
            Some(r) => r,
            None => return Ok(()),
        };

        let response = handler(&request);
        requests.lock().unwrap().push(request);

        match response {
            MockResponse::Silent => {
                // Never respond; hold the connection until the peer leaves.
                let mut buf = [0u8; 512];
                loop {
                    match tokio::time::timeout(Duration::from_secs(30), stream.read(&mut buf)).await
                    {
                        Err(_) => return Ok(()),
                        Ok(Err(_)) => return Ok(()),
                        Ok(Ok(0)) => return Ok(()),
                        Ok(Ok(_)) => continue,
                    }
                }
            }
            MockResponse::Full {
                status,
                headers,
                body,
            } => {
                let mut head = format!("HTTP/1.1 {} {}\r\n", status, status_reason(status));
                for (k, v) in &headers {
                    head.push_str(&format!("{}: {}\r\n", k, v));
                }
                head.push_str(&format!("content-length: {}\r\n", body.len()));
                head.push_str("connection: keep-alive\r\n\r\n");
                stream.write_all(head.as_bytes()).await?;
                stream.write_all(&body).await?;
            }
            MockResponse::Stream {
                status,
                headers,
                chunks,
                keep_open,
            } => {
                let mut head = format!("HTTP/1.1 {} {}\r\n", status, status_reason(status));
                for (k, v) in &headers {
                    head.push_str(&format!("{}: {}\r\n", k, v));
                }
                head.push_str("transfer-encoding: chunked\r\n\r\n");
                stream.write_all(head.as_bytes()).await?;
                for chunk in chunks {
                    if chunk.delay > Duration::ZERO {
                        tokio::time::sleep(chunk.delay).await;
                    }
                    let payload = chunk.data.as_bytes();
                    let frame = format!("{:x}\r\n", payload.len());
                    if stream.write_all(frame.as_bytes()).await.is_err() {
                        return Ok(());
                    }
                    if stream.write_all(payload).await.is_err() {
                        return Ok(());
                    }
                    if stream.write_all(b"\r\n").await.is_err() {
                        return Ok(());
                    }
                }
                if !keep_open {
                    if stream.write_all(b"0\r\n\r\n").await.is_err() {
                        return Ok(());
                    }
                } else {
                    // Hold the body open until the peer goes away.
                    let mut buf = [0u8; 512];
                    loop {
                        match tokio::time::timeout(Duration::from_secs(30), stream.read(&mut buf))
                            .await
                        {
                            Err(_) => return Ok(()),
                            Ok(Err(_)) => return Ok(()),
                            Ok(Ok(0)) => return Ok(()),
                            Ok(Ok(_)) => continue,
                        }
                    }
                }
            }
        }
    }
}

async fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<RecordedRequest>> {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];

    let head_end = loop {
        if let Some(pos) = find_double_crlf(&buf) {
            break pos;
        }
        let n = match tokio::time::timeout(Duration::from_secs(30), stream.read(&mut tmp)).await {
            Err(_) => return Ok(None),
            Ok(Err(_)) => return Ok(None),
            Ok(Ok(0)) => return Ok(None),
            Ok(Ok(n)) => n,
        };
        buf.extend_from_slice(&tmp[..n]);
    };

    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut body: Vec<u8> = buf[head_end + 4..].to_vec();
    while body.len() < content_length {
        let n = match tokio::time::timeout(Duration::from_secs(30), stream.read(&mut tmp)).await {
            Err(_) => return Ok(None),
            Ok(Err(_)) => return Ok(None),
            Ok(Ok(0)) => return Ok(None),
            Ok(Ok(n)) => n,
        };
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_length);
    let raw_body = body.clone();
    let body = maybe_decompress_request_body(&headers, body);

    Ok(Some(RecordedRequest {
        method,
        path,
        headers,
        body,
        raw_body,
    }))
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "X",
    }
}

// ---------------------------------------------------------------------------
// SSE payload builders (mirrors pi-ai test/openai-codex-stream.test.ts)
// ---------------------------------------------------------------------------

pub fn sse_line(event: serde_json::Value) -> String {
    format!("data: {}", event)
}

pub fn basic_completion_events(status: &str, end_turn: Option<bool>) -> Vec<String> {
    let terminal = if status == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    vec![
        sse_line(serde_json::json!({
            "type": "response.output_item.added",
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
        })),
        sse_line(serde_json::json!({
            "type": "response.content_part.added",
            "part": { "type": "output_text", "text": "" },
        })),
        sse_line(serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "Hello",
        })),
        sse_line(serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "output_text", "text": "Hello" }],
            },
        })),
        sse_line(serde_json::json!({
            "type": terminal,
            "response": {
                "status": status,
                "end_turn": end_turn,
                "incomplete_details": if status == "incomplete" {
                    serde_json::json!({ "reason": "max_output_tokens" })
                } else {
                    serde_json::Value::Null
                },
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                    "input_tokens_details": { "cached_tokens": 0 },
                },
            },
        })),
    ]
}

pub fn usage_events_with_tier(input: u64, output: u64, service_tier: Option<&str>) -> Vec<String> {
    let mut response = serde_json::json!({
        "status": "completed",
        "usage": {
            "input_tokens": input,
            "output_tokens": output,
            "total_tokens": input + output,
            "input_tokens_details": { "cached_tokens": 0 },
        },
    });
    if let Some(tier) = service_tier {
        response["service_tier"] = serde_json::json!(tier);
    }
    vec![
        sse_line(serde_json::json!({
            "type": "response.output_item.added",
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
        })),
        sse_line(serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "Hello",
        })),
        sse_line(serde_json::json!({
            "type": "response.completed",
            "response": response,
        })),
    ]
}

// ---------------------------------------------------------------------------
// WebSocket mock server (for the codex-websocket feature tests)
// ---------------------------------------------------------------------------

#[cfg(feature = "codex-websocket")]
#[allow(clippy::type_complexity)]
pub mod ws {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::net::{TcpListener, TcpStream};

    /// What the handler wants the mock server to do with the connection.
    #[derive(Debug, Clone)]
    pub enum WsReply {
        /// Send these JSON frames back (connection stays open for reuse).
        Frames(Vec<serde_json::Value>),
        /// Keep the connection open but send nothing (idle behaviour).
        Hang,
        /// Close the WebSocket connection (transport-level failure).
        Close,
    }

    /// Handler receiving (connection id, per-connection request index, path,
    /// headers, parsed frame) and returning what to send back.
    pub type WsHandler = Arc<
        dyn Fn(usize, usize, &str, &HashMap<String, String>, serde_json::Value) -> WsReply
            + Send
            + Sync,
    >;

    /// Plain HTTP handler used by the combined server for the SSE fallback
    /// requests that hit the same port.
    pub type HttpHandler =
        Arc<dyn Fn(&super::RecordedRequest) -> super::MockResponse + Send + Sync>;

    pub struct WsMockServer {
        pub addr: std::net::SocketAddr,
        requests: Arc<Mutex<Vec<WsRecordedRequest>>>,
        http_requests: Arc<Mutex<Vec<super::RecordedRequest>>>,
        handle: tokio::task::JoinHandle<()>,
    }

    #[derive(Debug, Clone)]
    pub struct WsRecordedRequest {
        pub connection: usize,
        pub path: String,
        pub headers: HashMap<String, String>,
        pub frame: serde_json::Value,
    }

    impl WsRecordedRequest {
        pub fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .get(&name.to_ascii_lowercase())
                .map(|s| s.as_str())
        }
    }

    impl WsMockServer {
        pub fn url(&self) -> String {
            // The real backend base URL is http(s)://; the client derives the
            // WebSocket URL from it, so the mock exposes the http URL too.
            format!("http://{}", self.addr)
        }

        pub fn recorded(&self) -> Vec<WsRecordedRequest> {
            self.requests.lock().unwrap().clone()
        }

        pub fn recorded_http(&self) -> Vec<super::RecordedRequest> {
            self.http_requests.lock().unwrap().clone()
        }

        pub async fn spawn(handler: WsHandler) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind ws mock");
            let addr = listener.local_addr().expect("ws addr");
            let requests: Arc<Mutex<Vec<WsRecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
            let reqs = requests.clone();
            let conns = Arc::new(AtomicUsize::new(0));
            let conns_handle = conns.clone();
            let handle = tokio::spawn(async move {
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(x) => x,
                        Err(_) => return,
                    };
                    let handler = handler.clone();
                    let reqs = reqs.clone();
                    let conns = conns_handle.clone();
                    tokio::spawn(async move {
                        let conn_id = conns.fetch_add(1, Ordering::SeqCst) + 1;
                        let _ = serve_ws(conn_id, stream, handler, reqs).await;
                    });
                }
            });
            Self {
                addr,
                requests,
                http_requests: Arc::new(Mutex::new(Vec::new())),
                handle,
            }
        }

        /// Combined server: answers WebSocket upgrades with `ws_handler` and
        /// plain HTTP (SSE fallback) requests with `http_handler` on the same
        /// port — mirrors the real backend where both share one URL.
        pub async fn spawn_combined(ws_handler: WsHandler, http_handler: HttpHandler) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind ws mock");
            let addr = listener.local_addr().expect("ws addr");
            let requests: Arc<Mutex<Vec<WsRecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
            let http_requests: Arc<Mutex<Vec<super::RecordedRequest>>> =
                Arc::new(Mutex::new(Vec::new()));
            let reqs = requests.clone();
            let http_reqs = http_requests.clone();
            let conns = Arc::new(AtomicUsize::new(0));
            let conns_handle = conns.clone();
            let handle = tokio::spawn(async move {
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(x) => x,
                        Err(_) => return,
                    };
                    let ws_handler = ws_handler.clone();
                    let http_handler = http_handler.clone();
                    let reqs = reqs.clone();
                    let http_reqs = http_reqs.clone();
                    let conns = conns_handle.clone();
                    tokio::spawn(async move {
                        let _ = serve_combined(
                            stream,
                            ws_handler,
                            http_handler,
                            reqs,
                            http_reqs,
                            conns,
                        )
                        .await;
                    });
                }
            });
            Self {
                addr,
                requests,
                http_requests,
                handle,
            }
        }

        /// Combined server whose WebSocket upgrades are accepted on TCP but
        /// never answered (handshake hangs until the client's connect
        /// timeout), while HTTP requests are served normally.
        pub async fn spawn_combined_pending_handshake(http_handler: HttpHandler) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind ws mock");
            let addr = listener.local_addr().expect("ws addr");
            let requests: Arc<Mutex<Vec<WsRecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
            let http_requests: Arc<Mutex<Vec<super::RecordedRequest>>> =
                Arc::new(Mutex::new(Vec::new()));
            let reqs = requests.clone();
            let http_reqs = http_requests.clone();
            let handle = tokio::spawn(async move {
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(x) => x,
                        Err(_) => return,
                    };
                    let http_handler = http_handler.clone();
                    let reqs = reqs.clone();
                    let http_reqs = http_reqs.clone();
                    tokio::spawn(async move {
                        let _ = serve_combined_pending(stream, http_handler, reqs, http_reqs).await;
                    });
                }
            });
            Self {
                addr,
                requests,
                http_requests,
                handle,
            }
        }

        /// Server that accepts TCP connections but never completes the
        /// WebSocket handshake (for connect-timeout tests).
        pub async fn spawn_pending_handshake() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind ws mock");
            let addr = listener.local_addr().expect("ws addr");
            let requests: Arc<Mutex<Vec<WsRecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
            let handle = tokio::spawn(async move {
                loop {
                    let (mut stream, _) = match listener.accept().await {
                        Ok(x) => x,
                        Err(_) => return,
                    };
                    tokio::spawn(async move {
                        use tokio::io::AsyncReadExt;
                        // Read the upgrade request, then never respond so the
                        // client's connect future hangs until its timeout.
                        let mut buf = [0u8; 8192];
                        let _ = stream.read(&mut buf).await;
                        futures::future::pending::<()>().await;
                    });
                }
            });
            Self {
                addr,
                requests,
                http_requests: Arc::new(Mutex::new(Vec::new())),
                handle,
            }
        }

        pub fn shutdown(self) {
            self.handle.abort();
        }
    }

    async fn serve_ws(
        conn_id: usize,
        stream: TcpStream,
        handler: WsHandler,
        requests: Arc<Mutex<Vec<WsRecordedRequest>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Capture the upgrade request's path/headers via the handshake
        // callback, then speak plain WebSocket frames.
        let captured: Arc<Mutex<Option<(String, HashMap<String, String>)>>> =
            Arc::new(Mutex::new(None));
        let cap = captured.clone();
        #[allow(clippy::result_large_err)]
        let callback = move |req: &http::Request<()>,
                             resp: tokio_tungstenite::tungstenite::handshake::server::Response|
              -> Result<
            tokio_tungstenite::tungstenite::handshake::server::Response,
            tokio_tungstenite::tungstenite::handshake::server::ErrorResponse,
        > {
            let mut path = req.uri().path().to_string();
            if let Some(q) = req.uri().query() {
                path.push('?');
                path.push_str(q);
            }
            let mut headers = HashMap::new();
            for (k, v) in req.headers() {
                headers.insert(
                    k.as_str().to_ascii_lowercase(),
                    v.to_str().unwrap_or_default().to_string(),
                );
            }
            *cap.lock().unwrap() = Some((path, headers));
            Ok(resp)
        };
        let ws_stream = tokio_tungstenite::accept_hdr_async(stream, callback).await?;
        let (path, headers) = captured.lock().unwrap().clone().unwrap_or_default();
        serve_accepted(conn_id, ws_stream, path, headers, handler, requests).await;
        Ok(())
    }

    async fn serve_accepted<S>(
        conn_id: usize,
        ws_stream: S,
        path: String,
        headers: HashMap<String, String>,
        handler: WsHandler,
        requests: Arc<Mutex<Vec<WsRecordedRequest>>>,
    ) where
        S: futures::Stream<
                Item = Result<
                    tokio_tungstenite::tungstenite::Message,
                    tokio_tungstenite::tungstenite::Error,
                >,
            > + futures::Sink<tokio_tungstenite::tungstenite::Message>
            + Unpin,
    {
        use futures::{SinkExt, StreamExt};
        let mut ws = ws_stream;
        let mut req_idx: usize = 0;
        // Serve multiple request frames on the same connection (the client
        // reuses the socket for cached sessions).
        loop {
            let frame = loop {
                match ws.next().await {
                    Some(Ok(tokio_tungstenite::tungstenite::Message::Text(t))) => {
                        break serde_json::from_str::<serde_json::Value>(&t).unwrap_or_default();
                    }
                    Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) | None => return,
                    Some(Ok(_)) => continue,
                    Some(Err(_)) => return,
                }
            };
            req_idx += 1;
            requests.lock().unwrap().push(WsRecordedRequest {
                connection: conn_id,
                path: path.clone(),
                headers: headers.clone(),
                frame: frame.clone(),
            });
            match handler(conn_id, req_idx, &path, &headers, frame) {
                WsReply::Frames(replies) => {
                    for reply in replies {
                        if ws
                            .send(tokio_tungstenite::tungstenite::Message::Text(
                                reply.to_string(),
                            ))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
                WsReply::Hang => {}
                WsReply::Close => {
                    let _ = ws
                        .send(tokio_tungstenite::tungstenite::Message::Close(None))
                        .await;
                    return;
                }
            }
        }
    }

    /// Sniff the first request on a connection: WebSocket upgrades go to the
    /// WS handler, everything else is treated as plain HTTP (SSE fallback).
    async fn serve_combined(
        stream: TcpStream,
        ws_handler: WsHandler,
        http_handler: HttpHandler,
        requests: Arc<Mutex<Vec<WsRecordedRequest>>>,
        http_requests: Arc<Mutex<Vec<super::RecordedRequest>>>,
        conns: Arc<AtomicUsize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (stream, buf) = peek_head(stream).await?;
        let head = String::from_utf8_lossy(&buf).to_string();
        if is_websocket_upgrade(&head) {
            let conn_id = conns.fetch_add(1, Ordering::SeqCst) + 1;
            // The upgrade request has no body, so `buf` is the complete head.
            let joined = std::io::Cursor::new(buf);
            let mut spliced = Spliced::new(stream, joined);
            let conn_id_clone = conn_id;
            let _ = serve_ws_from(&mut spliced, conn_id_clone, ws_handler, requests).await;
        } else {
            let (request, mut stream) = parse_http_head(&buf, stream).await?;
            let response = http_handler(&request);
            http_requests.lock().unwrap().push(request);
            write_http_response(&mut stream, response).await?;
        }
        Ok(())
    }

    /// WebSocket upgrades hang (never answered); HTTP requests are served.
    async fn serve_combined_pending(
        stream: TcpStream,
        http_handler: HttpHandler,
        requests: Arc<Mutex<Vec<WsRecordedRequest>>>,
        http_requests: Arc<Mutex<Vec<super::RecordedRequest>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (stream, buf) = peek_head(stream).await?;
        let head = String::from_utf8_lossy(&buf).to_string();
        if is_websocket_upgrade(&head) {
            // Hold the TCP connection open without completing the handshake.
            let _ = &requests;
            futures::future::pending::<()>().await;
        } else {
            let (request, mut stream) = parse_http_head(&buf, stream).await?;
            let response = http_handler(&request);
            http_requests.lock().unwrap().push(request);
            write_http_response(&mut stream, response).await?;
        }
        Ok(())
    }

    fn is_websocket_upgrade(head: &str) -> bool {
        head.to_ascii_lowercase().contains("upgrade: websocket")
    }

    /// Read until the request head (\r\n\r\n) is in the buffer and return
    /// ALL buffered bytes; the body may already be included.
    async fn peek_head(
        mut stream: TcpStream,
    ) -> Result<(TcpStream, Vec<u8>), Box<dyn std::error::Error>> {
        use tokio::io::AsyncReadExt;
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
            let n = stream.read(&mut tmp).await?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
        }
        Ok((stream, buf))
    }

    /// Parse the request from the buffered bytes (head + possibly part of the
    /// body), reading any remaining body bytes from the stream.
    async fn parse_http_head(
        buf: &[u8],
        mut stream: TcpStream,
    ) -> Result<(super::RecordedRequest, TcpStream), Box<dyn std::error::Error>> {
        use tokio::io::AsyncReadExt;
        let head_end = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|p| p + 4)
            .unwrap_or(buf.len());
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        let mut body: Vec<u8> = buf[head_end..].to_vec();
        let mut lines = head.split("\r\n");
        let request_line = lines.next().unwrap_or("");
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let mut headers = HashMap::new();
        for line in lines {
            if let Some((k, v)) = line.split_once(':') {
                headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
        }
        let content_length: usize = headers
            .get("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        while body.len() < content_length {
            let mut tmp = [0u8; 4096];
            let n = stream.read(&mut tmp).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&tmp[..n]);
        }
        body.truncate(content_length);
        let raw_body = body.clone();
        let body = super::maybe_decompress_request_body(&headers, body);
        Ok((
            super::RecordedRequest {
                method,
                path,
                headers,
                body,
                raw_body,
            },
            stream,
        ))
    }

    async fn write_http_response(
        stream: &mut TcpStream,
        response: super::MockResponse,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use tokio::io::AsyncWriteExt;
        if let super::MockResponse::Full {
            status,
            headers,
            body,
        } = response
        {
            let mut head = format!("HTTP/1.1 {} {}\r\n", status, super::status_reason(status));
            for (k, v) in &headers {
                head.push_str(&format!("{}: {}\r\n", k, v));
            }
            head.push_str(&format!("content-length: {}\r\n", body.len()));
            head.push_str("connection: keep-alive\r\n\r\n");
            stream.write_all(head.as_bytes()).await?;
            stream.write_all(&body).await?;
        }
        Ok(())
    }

    /// Wraps a TcpStream with a buffered head so the handshake parser sees the
    /// already-read bytes first.
    struct Spliced<R> {
        stream: R,
        prefix: std::io::Cursor<Vec<u8>>,
    }

    impl<R> Spliced<R> {
        fn new(stream: R, prefix: std::io::Cursor<Vec<u8>>) -> Self {
            Self { stream, prefix }
        }
    }

    impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for Spliced<R> {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            let prefix_len = self.prefix.get_ref().len();
            let pos = self.prefix.position() as usize;
            if pos < prefix_len {
                let n = std::cmp::min(buf.remaining(), prefix_len - pos);
                buf.put_slice(&self.prefix.get_ref()[pos..pos + n]);
                self.prefix.set_position((pos + n) as u64);
                return std::task::Poll::Ready(Ok(()));
            }
            std::pin::Pin::new(&mut self.stream).poll_read(cx, buf)
        }
    }

    impl<R: tokio::io::AsyncWrite + Unpin> tokio::io::AsyncWrite for Spliced<R> {
        fn poll_write(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
            buf: &[u8],
        ) -> std::task::Poll<std::io::Result<usize>> {
            std::pin::Pin::new(&mut self.stream).poll_write(cx, buf)
        }

        fn poll_flush(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::pin::Pin::new(&mut self.stream).poll_flush(cx)
        }

        fn poll_shutdown(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::pin::Pin::new(&mut self.stream).poll_shutdown(cx)
        }
    }

    /// Like `serve_ws` but reads the handshake from a buffered splice.
    async fn serve_ws_from<S>(
        spliced: &mut S,
        conn_id: usize,
        handler: WsHandler,
        requests: Arc<Mutex<Vec<WsRecordedRequest>>>,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        let captured: Arc<Mutex<Option<(String, HashMap<String, String>)>>> =
            Arc::new(Mutex::new(None));
        let cap = captured.clone();
        #[allow(clippy::result_large_err)]
        let callback = move |req: &http::Request<()>,
                             resp: tokio_tungstenite::tungstenite::handshake::server::Response|
              -> Result<
            tokio_tungstenite::tungstenite::handshake::server::Response,
            tokio_tungstenite::tungstenite::handshake::server::ErrorResponse,
        > {
            let mut path = req.uri().path().to_string();
            if let Some(q) = req.uri().query() {
                path.push('?');
                path.push_str(q);
            }
            let mut headers = HashMap::new();
            for (k, v) in req.headers() {
                headers.insert(
                    k.as_str().to_ascii_lowercase(),
                    v.to_str().unwrap_or_default().to_string(),
                );
            }
            *cap.lock().unwrap() = Some((path, headers));
            Ok(resp)
        };
        let ws_stream = tokio_tungstenite::accept_hdr_async(spliced, callback).await?;
        let (path, headers) = captured.lock().unwrap().clone().unwrap_or_default();
        serve_accepted(conn_id, ws_stream, path, headers, handler, requests).await;
        Ok(())
    }
}
