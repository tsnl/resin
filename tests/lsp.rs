use crossbeam_channel::{Receiver, unbounded};
use lsp_server::{Message, Notification, Request, Response};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    io::BufReader,
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    thread,
    time::{Duration, Instant},
};
use tempfile::TempDir;

struct Client {
    child: Child,
    input: ChildStdin,
    output: Receiver<Message>,
    inbox: VecDeque<Message>,
    next_id: i32,
}

impl Client {
    fn start(root: &Path, options: Value) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_resin"))
            .arg("--lsp")
            .arg(root)
            .current_dir(root.parent().unwrap())
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
        assert_eq!(
            initialize["capabilities"]["documentFormattingProvider"],
            true
        );
        assert_eq!(
            initialize["capabilities"]["completionProvider"]["triggerCharacters"],
            json!(["."])
        );
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
fn explicit_inference_updates_hover_after_edits() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("inference.resin"));
    for (version, initializer, expected) in [
        (1, "42", "long"),
        (2, "1 == 1", "bool"),
        (3, "\"text\"", "str"),
    ] {
        let source =
            format!("def answer() -> _ = {{ var value: _; value := {initializer}; value }};");
        if version == 1 {
            client.open(&uri, &source);
        } else {
            client.change(&uri, version, &source);
        }
        client.diagnostics(&uri, Some(version), false);
        let hover = client.request(
            "textDocument/hover",
            at(&uri, 0, source.rfind("value").unwrap() as u32),
        );
        assert!(
            hover.to_string().contains(&format!("value: {expected}")),
            "{hover}"
        );
    }
    client.stop();
}

#[test]
fn formatting_uses_current_buffers_and_returns_utf16_edits() {
    fn apply(source: &str, edits: &Value) -> String {
        fn offset(source: &str, position: &Value) -> usize {
            let line = position["line"].as_u64().unwrap() as usize;
            let column = position["character"].as_u64().unwrap() as usize;
            let start = if line == 0 {
                0
            } else {
                source.match_indices('\n').nth(line - 1).unwrap().0 + 1
            };
            let mut units = 0;
            for (byte, c) in source[start..].char_indices() {
                if units == column {
                    return start + byte;
                }
                units += c.len_utf16();
            }
            assert_eq!(units, column);
            source.len()
        }
        let mut result = source.to_owned();
        for edit in edits.as_array().unwrap().iter().rev() {
            let start = offset(source, &edit["range"]["start"]);
            let end = offset(source, &edit["range"]["end"]);
            result.replace_range(start..end, edit["newText"].as_str().unwrap());
        }
        result
    }
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = temp.path().join("format.resin");
    let file = uri(&path);
    let disk = "def main() = {};\n";
    std::fs::write(&path, disk).unwrap();
    let mut client = Client::start(temp.path(), json!({}));
    // Even clients requesting spaces receive the language's canonical hard tabs.
    let params =
        json!({"textDocument": {"uri": file}, "options": {"tabSize": 2, "insertSpaces": true}});
    client.open(&file, disk);
    let source = "/* 😀 */def main()={var xs=[1,2,];missing(xs);};\r\n";
    client.change(&file, 2, source);
    let edits = client.request("textDocument/formatting", params.clone());
    let expected =
        "/* 😀 */ def main() = {\n\tvar xs = [\n\t\t1,\n\t\t2,\n\t];\n\tmissing(xs);\n};\n";
    assert_eq!(
        edits[0]["range"]["start"],
        json!({"line": 0, "character": 8})
    );
    assert_eq!(apply(source, &edits), expected);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), disk);

    client.change(&file, 3, expected);
    client.change(&file, 2, "def stale( ="); // Older versions must not replace the buffer.
    assert_eq!(
        client.request("textDocument/formatting", params.clone()),
        json!([])
    );
    client.change(&file, 4, "def main()={var x=;};");
    assert_eq!(
        client.request("textDocument/formatting", params.clone()),
        Value::Null
    );
    client.change(&file, 5, "\r\n\t\r\n");
    assert_eq!(
        apply(
            "\r\n\t\r\n",
            &client.request("textDocument/formatting", params.clone())
        ),
        ""
    );
    // An edit ending immediately before LF must also replace the preceding CR.
    client.change(&file, 6, "def main() = {};\r\n");
    assert_eq!(
        apply(
            "def main() = {};\r\n",
            &client.request("textDocument/formatting", params.clone())
        ),
        disk
    );
    let malformed = client.response(
        "textDocument/formatting",
        json!({"textDocument": {"uri": file}}),
    );
    assert_eq!(malformed.error.unwrap().code, -32602);
    client.notify(
        "textDocument/didClose",
        json!({"textDocument": {"uri": file}}),
    );
    assert_eq!(
        client.request("textDocument/formatting", params),
        Value::Null
    );
    client.stop();
}

#[test]
fn dot_completion_updates_unsaved_receiver_types_and_uses_utf16_edits() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("fields.resin"));
    for (version, field, typed) in [(1, "count", ""), (2, "length", "le")] {
        let source =
            format!("def main () = {{ var value = {{ {field} = 1 }}; /* 😀 */ value.{typed}; }};");
        if version == 1 {
            client.open(&uri, &source);
        } else {
            client.change(&uri, version, &source);
        }
        client.diagnostics(&uri, Some(version), true);
        let start = source[..source.rfind('.').unwrap() + 1]
            .encode_utf16()
            .count() as u32;
        let mut params = at(&uri, 0, start + typed.len() as u32);
        params["context"] = json!({"triggerKind": 2, "triggerCharacter": "."});
        let completion = client.request("textDocument/completion", params);
        let items = completion["items"].as_array().unwrap();
        assert_eq!(items.len(), 1, "{completion}");
        assert_eq!(items[0]["label"], field);
        assert_eq!(items[0]["kind"], 5); // CompletionItemKind::FIELD
        assert_eq!(
            items[0]["textEdit"],
            json!({
                "range": {"start": {"line": 0, "character": start}, "end": {"line": 0, "character": start + typed.len() as u32}},
                "newText": field
            })
        );
    }
    client.stop();
}

#[test]
fn string_completions_distinguish_literal_views_and_owned_constructors() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("strings.resin"));
    for (version, expression, labels) in [
        (1, "\"text\".", vec!["data", "length", "at"]),
        (2, "String.", vec!["from_bytes", "from_str"]),
    ] {
        let source = format!("def main() = {{ {expression}; }};");
        if version == 1 {
            client.open(&uri, &source);
        } else {
            client.change(&uri, version, &source);
        }
        client.diagnostics(&uri, Some(version), true);
        let column = source.rfind('.').unwrap() as u32 + 1;
        let completion = client.request("textDocument/completion", at(&uri, 0, column));
        let items = completion["items"].as_array().unwrap();
        let actual: Vec<_> = items
            .iter()
            .map(|item| item["label"].as_str().unwrap())
            .collect();
        assert_eq!(actual, labels, "{completion}");
    }
    client.stop();
}

#[test]
fn dot_completion_sorts_fields_before_methods() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("members.resin"));
    let source = "struct Record { zebra: int, middle: int };\n\
        impl Record {\n\
            def beta(self: Ptr<Record>) = {};\n\
            def alpha(self: Ptr<Record>) = {};\n\
        }\n\
        def main() = { var value = Record { zebra = 1, middle = 2 }; value.; };";
    client.open(&uri, source);
    client.diagnostics(&uri, Some(1), true);
    let column = source.lines().last().unwrap().rfind('.').unwrap() as u32 + 1;
    let completion = client.request("textDocument/completion", at(&uri, 5, column));
    let items = completion["items"].as_array().unwrap();
    let labels = |items: &[Value]| {
        items
            .iter()
            .map(|item| item["label"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    let expected = ["middle", "zebra", "alpha", "beta"];
    assert_eq!(labels(items), expected);
    assert!(items[..2].iter().all(|item| item["kind"] == 5)); // FIELD
    assert!(items[2..].iter().all(|item| item["kind"] == 3)); // FUNCTION

    // Clients use sortText instead of the response's array order or labels.
    let mut sorted = items.clone();
    sorted.reverse();
    sorted.sort_by(|a, b| {
        a["sortText"]
            .as_str()
            .unwrap()
            .cmp(b["sortText"].as_str().unwrap())
    });
    assert_eq!(labels(&sorted), expected);
    client.stop();
}

#[test]
fn unsaved_unicode_buffers_support_features_edits_and_clean_shutdown() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let root = temp.path();
    let library = root.join("lib.resin");
    std::fs::write(
        &library,
        "export { answer }; def answer () -> int = { 1 }; def private () -> int = { 2 };",
    )
    .unwrap();
    let mut client = Client::start(root, Value::Null);
    let main_uri = uri(&root.join("main.resin"));
    let lib_uri = uri(&resin_source::normalize_path(&library).unwrap());
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
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let stdlib = temp.path().join("standard");
    std::fs::create_dir(&stdlib).unwrap();
    std::fs::write(
        stdlib.join("custom.resin"),
        "export { standard }; def standard () -> int = { 1 };",
    )
    .unwrap();
    let mut client = Client::start(temp.path(), json!({"stdlibPath": "standard"}));
    let uri = uri(&temp.path().join("main.resin"));
    client.open(
        &uri,
        "import { \"$/std/custom.resin\" }; def main () -> int = { standard() };",
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

#[test]
fn holes_do_not_block_later_features_and_repair_clears_diagnostics() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("holes.resin"));
    let broken = "def main() = { var missing = ; var value = { count = 1 }; value.count; };";
    client.open(&uri, broken);
    client.diagnostics(&uri, Some(1), true);
    let end = broken.rfind("count").unwrap() as u32 + 2;
    let completion = client.request("textDocument/completion", at(&uri, 0, end));
    assert_eq!(completion["items"][0]["label"], "count");
    let fixed = broken.replace("missing = ;", "missing = 2;");
    client.change(&uri, 2, &fixed);
    client.diagnostics(&uri, Some(2), false);
    let hover = client.request(
        "textDocument/hover",
        at(&uri, 0, fixed.find("missing").unwrap() as u32),
    );
    assert!(
        hover["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("missing: long")
    );
    let unknown = "def main() = { var value = ; value.count; };";
    client.change(&uri, 3, unknown);
    client.diagnostics(&uri, Some(3), true);
    let completion = client.request(
        "textDocument/completion",
        at(&uri, 0, unknown.rfind("count").unwrap() as u32 + 2),
    );
    assert!(completion["items"].as_array().unwrap().is_empty());
    client.stop();
}
