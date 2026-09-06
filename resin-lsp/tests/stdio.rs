use crossbeam_channel::{Receiver, unbounded};
use lsp_server::{Message, Notification, Request, Response};
use resin::toolchain::TempDir;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    io::BufReader,
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Client {
    child: Child,
    input: ChildStdin,
    output: Receiver<Message>,
    inbox: VecDeque<Message>,
    next_id: i32,
}

impl Client {
    fn start(root: &Path, options: Value) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_resin-lsp"))
            .arg("--stdio")
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let (sender, output) = unbounded();
        thread::spawn(move || {
            while let Ok(Some(message)) = Message::read(&mut stdout) {
                if sender.send(message).is_err() {
                    break;
                }
            }
        });
        let mut client = Self {
            child,
            input,
            output,
            inbox: VecDeque::new(),
            next_id: 1,
        };
        let initialize = client.request(
            "initialize",
            json!({
                "processId": null, "rootUri": uri(root), "capabilities": {
                    "general": {"positionEncodings": ["utf-16"]},
                    "workspace": {"didChangeWatchedFiles": {"dynamicRegistration": true}}
                }, "initializationOptions": options
            }),
        );
        assert_eq!(initialize["capabilities"]["positionEncoding"], "utf-16");
        assert_eq!(initialize["capabilities"]["textDocumentSync"]["change"], 1);
        client.notify("initialized", json!({}));
        client
    }

    fn send(&mut self, message: Message) {
        message.write(&mut self.input).unwrap();
    }
    fn notify(&mut self, method: &str, params: Value) {
        self.send(Message::Notification(Notification::new(
            method.into(),
            params,
        )));
    }
    fn response(&mut self, method: &str, params: Value) -> Response {
        let id = self.next_id;
        self.next_id += 1;
        self.send(Message::Request(Request::new(
            id.into(),
            method.into(),
            params,
        )));
        match self.wait_for(|m| matches!(m, Message::Response(r) if r.id == id.into())) {
            Message::Response(response) => response,
            _ => unreachable!(),
        }
    }
    fn request(&mut self, method: &str, params: Value) -> Value {
        let response = self.response(method, params);
        assert!(response.error.is_none(), "{response:?}");
        response.result.unwrap_or(Value::Null)
    }
    fn wait_for(&mut self, predicate: impl Fn(&Message) -> bool) -> Message {
        if let Some(index) = self.inbox.iter().position(&predicate) {
            return self.inbox.remove(index).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let message = self
                .output
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .expect("timed out waiting for LSP message");
            if let Message::Request(request) = message {
                assert_eq!(request.method, "client/registerCapability");
                self.send(Message::Response(Response::new_ok(request.id, Value::Null)));
                continue;
            }
            if predicate(&message) {
                return message;
            }
            self.inbox.push_back(message);
        }
    }
    fn diagnostics(&mut self, uri: &str, version: Option<i32>, has_errors: bool) -> Value {
        let message = self.wait_for(|message| matches!(message, Message::Notification(n) if n.method == "textDocument/publishDiagnostics" && n.params["uri"] == uri && n.params["version"] == json!(version) && n.params["diagnostics"].as_array().unwrap().is_empty() != has_errors));
        let Message::Notification(notification) = message else {
            unreachable!()
        };
        notification.params["diagnostics"].clone()
    }
    fn open(&mut self, uri: &str, source: &str) {
        self.notify("textDocument/didOpen", json!({"textDocument": {"uri": uri, "languageId": "resin", "version": 1, "text": source}}));
    }
    fn change(&mut self, uri: &str, version: i32, source: &str) {
        self.notify("textDocument/didChange", json!({"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": source}]}));
    }
    fn stop(mut self) {
        assert_eq!(self.request("shutdown", Value::Null), Value::Null);
        self.notify("exit", Value::Null);
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            assert!(Instant::now() < deadline, "server did not exit");
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn uri(path: &Path) -> String {
    url::Url::from_file_path(path).unwrap().into()
}
fn at(uri: &str, line: u32, character: u32) -> Value {
    json!({"textDocument": {"uri": uri}, "position": {"line": line, "character": character}})
}

#[test]
fn unsaved_unicode_buffers_support_features_edits_and_clean_shutdown() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("new.resin"));
    let source = "def main (argument: int) -> int = {\r\n  // 😀\r\n  var value = argument + 1;\r\n  /* 😀 */ value\r\n};\r\n";
    client.open(&uri, source);
    client.diagnostics(&uri, Some(1), false);
    let hover = client.request("textDocument/hover", at(&uri, 3, 12));
    assert!(
        hover["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("value: int")
    );
    assert_eq!(hover["range"]["start"], json!({"line": 3, "character": 11}));
    let definition = client.request("textDocument/definition", at(&uri, 3, 12));
    assert_eq!(definition["uri"], uri);
    assert_eq!(
        definition["range"]["start"],
        json!({"line": 2, "character": 6})
    );
    let completion = client.request("textDocument/completion", at(&uri, 3, 14));
    let item = completion["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["label"] == "value")
        .unwrap();
    assert_eq!(
        item["textEdit"]["range"],
        json!({"start": {"line": 3, "character": 11}, "end": {"line": 3, "character": 16}})
    );
    assert_eq!(
        client
            .response("textDocument/hover", at(&uri, 3, 6))
            .error
            .unwrap()
            .code,
        -32602,
        "split surrogate must be rejected"
    );
    assert_eq!(
        client
            .response("unknown/request", json!({}))
            .error
            .unwrap()
            .code,
        -32601
    );

    client.change(&uri, 2, &source.replace("value =", "renamed ="));
    let errors = client.diagnostics(&uri, Some(2), true);
    assert_eq!(
        errors[0]["range"]["start"],
        json!({"line": 3, "character": 11})
    );
    client.change(&uri, 1, source); // Must not replace the newer buffer.
    assert_eq!(
        client.request("textDocument/definition", at(&uri, 3, 12)),
        Value::Null
    );
    client.change(&uri, 3, "def main (argument: int) -> int = { arg");
    client.diagnostics(&uri, Some(3), true);
    let completion = client.request("textDocument/completion", at(&uri, 0, 37));
    assert!(
        completion["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["label"] == "argument")
    );
    client.notify(
        "textDocument/didClose",
        json!({"textDocument": {"uri": uri}}),
    );
    client.diagnostics(&uri, None, false);
    client.stop();
}

#[test]
fn dependency_overlays_close_and_disk_changes_refresh_consumers() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let root = temp.path();
    let library = root.join("lib.resin");
    std::fs::write(
        &library,
        "export { answer }; def answer () -> int = { 1 }; def private () -> int = { 2 };",
    )
    .unwrap();
    let mut client = Client::start(root, Value::Null);
    let main_uri = uri(&root.join("main.resin"));
    let lib_uri = uri(&library);
    let source = "import { \"lib.resin\" }; def main () -> int = { answer() };";
    let column = source.rfind("answer").unwrap() as u32;
    client.open(&main_uri, source);
    client.diagnostics(&main_uri, Some(1), false);
    assert_eq!(
        client.request("textDocument/definition", at(&main_uri, 0, column))["uri"],
        lib_uri
    );
    let completions = client.request("textDocument/completion", at(&main_uri, 0, column));
    assert!(
        !completions["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["label"] == "private")
    );

    client.open(
        &lib_uri,
        "export { renamed }; def renamed () -> int = { 2 };",
    );
    client.diagnostics(&main_uri, Some(1), true);
    assert_eq!(
        client.request("textDocument/definition", at(&main_uri, 0, column)),
        Value::Null
    );
    client.notify(
        "textDocument/didClose",
        json!({"textDocument": {"uri": lib_uri}}),
    );
    client.diagnostics(&main_uri, Some(1), false);
    assert_eq!(
        client.request("textDocument/definition", at(&main_uri, 0, column))["uri"],
        lib_uri
    );

    std::fs::write(&library, "export { answer }; def answer () -> int = { + };").unwrap();
    client.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": lib_uri, "type": 2}]}),
    );
    client.diagnostics(&lib_uri, None, true);
    std::fs::write(&library, "export { answer }; def answer () -> int = { 3 };").unwrap();
    client.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": lib_uri, "type": 2}]}),
    );
    client.diagnostics(&lib_uri, None, false);
    let hover = client.request("textDocument/hover", at(&main_uri, 0, column));
    assert!(
        hover["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("def answer () -> int")
    );
    client.stop();
}

#[test]
fn stdlib_override_and_rapid_versions_use_the_latest_snapshot() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let stdlib = temp.path().join("standard");
    std::fs::create_dir(&stdlib).unwrap();
    std::fs::write(
        stdlib.join("custom.resin"),
        "export { standard }; def standard () -> int = { 1 };",
    )
    .unwrap();
    let mut client = Client::start(temp.path(), json!({"stdlibPath": stdlib}));
    let uri = uri(&temp.path().join("main.resin"));
    client.open(
        &uri,
        "import { \"std/custom.resin\" }; def main () -> int = { standard() };",
    );
    client.diagnostics(&uri, Some(1), false);
    for version in 2..50 {
        client.change(
            &uri,
            version,
            &format!("def main () -> int = {{ var value{version} = missing; value{version} }};"),
        );
    }
    let final_source = "def main () -> int = { var final = 1; final };";
    client.change(&uri, 50, final_source);
    client.diagnostics(&uri, Some(50), false);
    let hover = client.request(
        "textDocument/hover",
        at(&uri, 0, final_source.rfind("final").unwrap() as u32),
    );
    assert!(
        hover["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("final: int")
    );
    client.stop();
}
