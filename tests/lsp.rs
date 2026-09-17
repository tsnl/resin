#[allow(dead_code)]
mod support;

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
    service: support::service::Service,
    child: Child,
    input: ChildStdin,
    output: Receiver<Message>,
    inbox: VecDeque<Message>,
    registrations: Vec<Value>,
    next_id: i32,
}

impl Client {
    fn start(root: &Path, options: Value) -> Self {
        Self::start_with_library_root(root, options, None)
    }

    fn start_with_library_root(root: &Path, options: Value, library_root: Option<&Path>) -> Self {
        Self::start_with_compiler(root, options, library_root, None)
    }

    fn start_with_compiler(
        root: &Path,
        options: Value,
        library_root: Option<&Path>,
        compiler: Option<&Path>,
    ) -> Self {
        let service = support::service::Service::configured(|config, environment| {
            if let Some(library_root) = library_root {
                config.library_root = root.parent().unwrap().join(library_root);
            }
            if let Some(compiler) = compiler {
                environment
                    .variables
                    .insert("CC".into(), compiler.as_os_str().to_owned());
            }
        });
        let mut command = service.command();
        command
            .arg("--lsp")
            .arg(root)
            .current_dir(root.parent().unwrap())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command.spawn().unwrap();
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
            service,
            child,
            input,
            output,
            inbox: VecDeque::new(),
            registrations: Vec::new(),
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
                self.registrations.push(request.params.clone());
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
            format!("fn answer() -> _  {{ let mut value: _; value = {initializer}; value }}");
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
    let disk = "fn main() {}\n";
    std::fs::write(&path, disk).unwrap();
    let mut client = Client::start(temp.path(), json!({}));
    // Even clients requesting spaces receive the language's canonical hard tabs.
    let params =
        json!({"textDocument": {"uri": file}, "options": {"tabSize": 2, "insertSpaces": true}});
    client.open(&file, disk);
    let source = "/* 😀 */fn main(){let mut xs=[1,2,];missing(xs);}\r\n";
    client.change(&file, 2, source);
    let edits = client.request("textDocument/formatting", params.clone());
    let expected =
        "/* 😀 */ fn main() {\n\tlet mut xs = [\n\t\t1,\n\t\t2,\n\t];\n\tmissing(xs);\n}\n";
    assert_eq!(
        edits[0]["range"]["start"],
        json!({"line": 0, "character": 8})
    );
    assert_eq!(apply(source, &edits), expected);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), disk);

    client.change(&file, 3, expected);
    client.change(&file, 2, "fn stale( ="); // Older versions must not replace the buffer.
    assert_eq!(
        client.request("textDocument/formatting", params.clone()),
        json!([])
    );
    client.change(&file, 4, "fn main(){let mut x=;}");
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
    client.change(&file, 6, "fn main() {}\r\n");
    assert_eq!(
        apply(
            "fn main() {}\r\n",
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
fn gradient_dot_completion_survives_edits_before_kernel_statements() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let library = Path::new(env!("CARGO_MANIFEST_DIR")).join("resin");
    let mut client = Client::start_with_library_root(temp.path(), Value::Null, Some(&library));
    let uri = uri(&temp.path().join("gradient.resin"));
    let original = include_str!("../examples/gradient.resin");
    client.open(&uri, original);
    client.diagnostics(&uri, Some(1), false);
    let positions = [
        original.find("\tif (index").unwrap(),
        original.find("\n}\n\nfn main").unwrap() + 1,
    ];
    for (index, offset) in positions.into_iter().enumerate() {
        let mut source = original.to_owned();
        source.insert_str(offset, "\troot.\n");
        let version = index as i32 + 2;
        client.change(&uri, version, &source);
        client.diagnostics(&uri, Some(version), true);
        let line = original[..offset].lines().count() as u32;
        let mut params = at(&uri, line, 6);
        params["context"] = json!({"triggerKind": 2, "triggerCharacter": "."});
        let completion = client.request("textDocument/completion", params);
        let fields = completion["items"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["kind"] == 5)
            .collect::<Vec<_>>();
        assert_eq!(
            fields
                .iter()
                .map(|item| item["label"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["height", "pixels", "width"]
        );
        for item in fields {
            assert_eq!(
                item["textEdit"]["range"],
                json!({
                    "start": {"line": line, "character": 6},
                    "end": {"line": line, "character": 6}
                })
            );
        }
    }
    client.change(&uri, 4, original);
    client.diagnostics(&uri, Some(4), false);
    client.stop();
}

#[test]
fn dot_completion_updates_unsaved_receiver_types_and_uses_utf16_edits() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("fields.resin"));
    for (version, field, typed) in [(1, "count", ""), (2, "length", "le")] {
        let source = format!(
            "struct Field {{ {field}: long, }} fn main ()  {{ let mut value = Field {{ {field} = 1 }}; /* 😀 */ value.{typed}; }}"
        );
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
fn string_completions_distinguish_fields_operations_and_free_constructors() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("strings.resin"));
    for (version, expression, required, absent) in [
        (1, "\"text\".", vec!["data", "length"], vec!["at"]),
        (
            2,
            "\"text\":",
            vec!["at", "print", "bytes"],
            vec!["data", "length"],
        ),
        (
            3,
            "string_",
            vec!["string_from_bytes", "string_from_str"],
            vec!["String"],
        ),
    ] {
        let source = format!("import {{ \"$/string.resin\" }}; fn main() {{ {expression}; }}");
        if version == 1 {
            client.open(&uri, &source);
        } else {
            client.change(&uri, version, &source);
        }
        client.diagnostics(&uri, Some(version), true);
        let column = (source.rfind(expression).unwrap() + expression.len()) as u32;
        let completion = client.request("textDocument/completion", at(&uri, 0, column));
        let labels: Vec<_> = completion["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["label"].as_str().unwrap())
            .collect();
        for name in required {
            assert!(labels.contains(&name), "{completion}");
        }
        for name in absent {
            assert!(!labels.contains(&name), "{completion}");
        }
    }
    client.stop();
}

#[test]
fn dot_completes_fields_and_colon_completes_visible_operations() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("members.resin"));
    for (version, separator, expected, kind) in [
        (1, ".", vec!["middle", "zebra"], 5),
        (2, ":", vec!["alpha", "beta"], 3),
    ] {
        let source = format!(
            "struct Record {{ zebra: int, middle: int }}\nfn beta(value: Ref<Record>) {{}}\nfn alpha(value: Ref<Record>) {{}}\nfn main() {{ let value = Record {{ zebra = 1, middle = 2 }}; value{separator}; }}"
        );
        if version == 1 {
            client.open(&uri, &source);
        } else {
            client.change(&uri, version, &source);
        }
        client.diagnostics(&uri, Some(version), true);
        let column = source.lines().last().unwrap().rfind(separator).unwrap() as u32 + 1;
        let completion = client.request("textDocument/completion", at(&uri, 3, column));
        let items = completion["items"].as_array().unwrap();
        let labels = |items: &[Value]| {
            items
                .iter()
                .map(|item| item["label"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(labels(items), expected);
        assert!(items.iter().all(|item| item["kind"] == kind));
        let mut sorted = items.clone();
        sorted.reverse();
        sorted.sort_by(|a, b| {
            a["sortText"]
                .as_str()
                .unwrap()
                .cmp(b["sortText"].as_str().unwrap())
        });
        assert_eq!(labels(&sorted), expected);
    }
    client.stop();
}

#[test]
fn unsaved_unicode_buffers_support_features_edits_and_clean_shutdown() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("new.resin"));
    let source = "fn main (argument: int) -> int  {\r\n  // 😀\r\n  let mut value = argument + 1;\r\n  /* 😀 */ value\r\n}\r\n";
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
        json!({"line": 2, "character": 10})
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
    let incomplete = "fn main (argument: int) -> int { arg";
    client.change(&uri, 3, incomplete);
    client.diagnostics(&uri, Some(3), true);
    let completion = client.request(
        "textDocument/completion",
        at(&uri, 0, incomplete.len() as u32),
    );
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
        "export { answer }; fn answer () -> int  { 1 } fn private () -> int  { 2 }",
    )
    .unwrap();
    let mut client = Client::start(root, Value::Null);
    let main_uri = uri(&root.join("main.resin"));
    let lib_uri = uri(&resin_source::normalize_path(&library).unwrap());
    let source = "import { \"lib.resin\" }; fn main () -> int  { answer() }";
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

    client.open(&lib_uri, "export { renamed }; fn renamed () -> int  { 2 }");
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

    std::fs::write(&library, "export { answer }; fn answer () -> int  { + }").unwrap();
    client.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": lib_uri, "type": 2}]}),
    );
    client.diagnostics(&lib_uri, None, true);
    std::fs::write(&library, "export { answer }; fn answer () -> int  { 3 }").unwrap();
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
            .contains("fn answer () -> int")
    );
    client.stop();
}

#[test]
fn local_header_creation_and_deletion_refresh_unchanged_editor_inputs() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    let include = temp.path().join("include");
    std::fs::create_dir(&project).unwrap();
    std::fs::create_dir(&include).unwrap();
    let mut client = Client::start(&project, json!({"includeRoots": ["../include"]}));
    let main_uri = uri(&project.join("main.resin"));
    let source = "extern { \"./fixture\": {} }; fn main()  {}";
    client.open(&main_uri, source);
    client.diagnostics(&main_uri, Some(1), true);
    let watchers: Vec<_> = client
        .registrations
        .iter()
        .flat_map(|registration| registration["registrations"].as_array().unwrap())
        .flat_map(|registration| {
            registration["registerOptions"]["watchers"]
                .as_array()
                .unwrap()
        })
        .collect();
    assert!(
        watchers
            .iter()
            .any(|watcher| watcher["globPattern"] == "**/*")
    );
    let watched_include = url::Url::from_directory_path(std::fs::canonicalize(&include).unwrap())
        .unwrap()
        .to_file_path()
        .unwrap();
    let include_pattern = watched_include
        .join("**")
        .join("*")
        .to_string_lossy()
        .replace('\\', "/");
    assert!(
        watchers
            .iter()
            .any(|watcher| watcher["globPattern"] == include_pattern)
    );

    for header in [project.join("fixture"), include.join("fixture")] {
        std::fs::write(&header, "/* complete local header */\n").unwrap();
        client.notify(
            "workspace/didChangeWatchedFiles",
            json!({"changes": [{"uri": uri(&header), "type": 1}]}),
        );
        client.diagnostics(&main_uri, Some(1), false);
        std::fs::remove_file(&header).unwrap();
        client.notify(
            "workspace/didChangeWatchedFiles",
            json!({"changes": [{"uri": uri(&header), "type": 3}]}),
        );
        client.diagnostics(&main_uri, Some(1), true);
    }
    client.stop();
}

#[test]
fn managed_library_roots_are_selected_by_server_configuration() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    std::fs::create_dir(&project).unwrap();
    for (directory, function) in [("first", "first_value"), ("second", "second_value")] {
        let library = temp.path().join(directory);
        std::fs::create_dir_all(library.join("math")).unwrap();
        std::fs::write(
            library.join("math/value.resin"),
            format!("export {{ {function} }}; fn {function}() -> int  {{ 42 }}"),
        )
        .unwrap();
        let mut client = Client::start_with_library_root(&project, Value::Null, Some(&library));
        let uri = uri(&project.join("main.resin"));
        client.open(
            &uri,
            &format!("import {{ \"$/math/value.resin\" }}; fn main() -> int  {{ {function}() }}"),
        );
        client.diagnostics(&uri, Some(1), false);
    }
}

#[test]
fn library_root_override_and_rapid_versions_use_the_latest_snapshot() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let library_root = temp.path().join("standard");
    std::fs::create_dir(&library_root).unwrap();
    std::fs::write(
        library_root.join("custom.resin"),
        "export { standard }; fn standard () -> int  { 1 }",
    )
    .unwrap();
    let mut client = Client::start_with_library_root(temp.path(), Value::Null, Some(&library_root));
    let uri = uri(&temp.path().join("main.resin"));
    client.open(
        &uri,
        "import { \"$/custom.resin\" }; fn main () -> int  { standard() }",
    );
    client.diagnostics(&uri, Some(1), false);
    for version in 2..50 {
        client.change(
            &uri,
            version,
            &format!("fn main () -> int  {{ let mut value{version} = missing; value{version} }}"),
        );
    }
    let final_source = "fn main () -> int  { let mut final = 1; final }";
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
    let broken = "struct FieldsCount<T0> { count: T0, } fn main()  { let mut missing = ; let mut value = FieldsCount<_> { count = 1 }; value.count; }";
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
    let unknown = "fn main()  { let mut value = ; value.count; }";
    client.change(&uri, 3, unknown);
    client.diagnostics(&uri, Some(3), true);
    let completion = client.request(
        "textDocument/completion",
        at(&uri, 0, unknown.rfind("count").unwrap() as u32 + 2),
    );
    assert!(completion["items"].as_array().unwrap().is_empty());
    client.stop();
}

#[test]
fn editor_builds_use_disk_and_preserve_dirty_buffers_across_saves() {
    let temp = TempDir::new().unwrap();
    let main_path = temp.path().join("main.resin");
    let helper_path = temp.path().join("helper.resin");
    let main = "export { main }; import { \"helper.resin\" }; fn main() -> int  { value() }";
    let helper = "export { value }; fn value() -> int  { 7 }";
    std::fs::write(&main_path, main).unwrap();
    std::fs::write(&helper_path, helper).unwrap();
    let main_uri = uri(&main_path);
    let helper_uri = uri(&helper_path);
    let mut client = Client::start(temp.path(), Value::Null);
    client.open(&main_uri, main);
    let dirty = "export { value }; fn value() -> bool  { 1 == 1 }";
    client.open(&helper_uri, dirty);
    client.diagnostics(&main_uri, Some(1), true);
    let output = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    let result = client.request(
        "workspace/executeCommand",
        json!({
            "command": "resin.build", "arguments": [{
                "uri": main_uri, "destination": output, "profile": "debug"
            }]
        }),
    );
    assert_eq!(result["outputUri"], uri(&output));
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(7));
    assert_eq!(std::fs::read_to_string(&helper_path).unwrap(), helper);

    // Saving the main file must preserve the helper's unsaved bool signature.
    client.notify(
        "textDocument/didSave",
        json!({"textDocument": {"uri": main_uri}}),
    );
    let hover = client.request(
        "textDocument/hover",
        at(&helper_uri, 0, dirty.find("value()").unwrap() as u32),
    );
    assert!(
        hover["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("bool")
    );
    client.notify(
        "textDocument/didClose",
        json!({"textDocument": {"uri": helper_uri}}),
    );
    client.diagnostics(&main_uri, Some(1), false);
    client.stop();
}

#[test]
fn editor_build_destinations_are_required_and_cannot_replace_inputs() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("main.resin");
    let source = "export { main }; fn main() -> int  { 0 }";
    std::fs::write(&path, source).unwrap();
    let source_uri = uri(&path);
    let mut client = Client::start(temp.path(), Value::Null);
    let missing = client.response(
        "workspace/executeCommand",
        json!({
            "command": "resin.build", "arguments": [{"uri": source_uri}]
        }),
    );
    assert!(missing.error.is_some());
    let overwrite = client.response(
        "workspace/executeCommand",
        json!({
            "command": "resin.build", "arguments": [{"uri": source_uri, "destination": path}]
        }),
    );
    assert!(overwrite.error.unwrap().message.contains("overwrite"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), source);
    client.stop();
}

#[cfg(unix)]
#[test]
fn editor_queries_and_cancellation_stay_responsive_during_native_work() {
    use std::os::unix::fs::PermissionsExt;
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("main.resin");
    let source = "export { main }; fn main() -> int  { 7 }";
    std::fs::write(&path, source).unwrap();
    let compiler = temp.path().join("compiler");
    let marker = temp.path().join("compiler-started");
    let release = temp.path().join("release");
    let real = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
    // Positional arguments and single-quoted paths keep the test wrapper literal.
    fn quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
    let script = format!(
        "#!/bin/sh\nfor argument in \"$@\"; do\n if [ \"$argument\" = -E ]; then exec {} \"$@\"; fi\ndone\nprintf '%s' \"$$\" > {}\nwhile [ ! -e {} ]; do sleep 0.01; done\nexec {} \"$@\"\n",
        quote(&real.to_string_lossy()),
        quote(&marker.to_string_lossy()),
        quote(&release.to_string_lossy()),
        quote(&real.to_string_lossy())
    );
    std::fs::write(&compiler, script).unwrap();
    std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o755)).unwrap();
    struct Release(std::path::PathBuf);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = std::fs::write(&self.0, "");
        }
    }
    let _release = Release(release);
    let mut client = Client::start_with_compiler(temp.path(), Value::Null, None, Some(&compiler));
    let source_uri = uri(&path);
    client.open(&source_uri, source);
    client.diagnostics(&source_uri, Some(1), false);
    let output = temp.path().join("program");
    client.send(Message::Request(Request::new(
        900.into(),
        "workspace/executeCommand".into(),
        json!({
            "command": "resin.build", "arguments": [{"uri": source_uri, "destination": output}]
        }),
    )));
    let deadline = Instant::now() + Duration::from_secs(10);
    while !marker.exists() {
        assert!(Instant::now() < deadline, "native compiler did not start");
        thread::sleep(Duration::from_millis(10));
    }
    let hover = client.request(
        "textDocument/hover",
        at(&source_uri, 0, source.find("main()").unwrap() as u32),
    );
    assert!(hover["contents"]["value"].as_str().unwrap().contains("int"));
    client.notify("$/cancelRequest", json!({"id": 900}));
    let cancelled = client.wait_for(
        |message| matches!(message, Message::Response(response) if response.id == 900.into()),
    );
    let Message::Response(cancelled) = cancelled else {
        unreachable!()
    };
    assert_eq!(
        cancelled.error.unwrap().code,
        lsp_server::ErrorCode::RequestCanceled as i32
    );
    assert!(!output.exists());
    client.stop();
    let pid = std::fs::read_to_string(marker).unwrap();
    let alive = Command::new("sh")
        .args(["-c", "kill -0 \"$1\" 2>/dev/null", "check", &pid])
        .status()
        .unwrap()
        .success();
    assert!(!alive, "compiler survived LSP shutdown");
}

#[test]
fn missing_parent_imports_recover_after_creation_deletion_and_alias_changes() {
    let temp = TempDir::new().unwrap();
    let child = temp.path().join("child");
    std::fs::create_dir(&child).unwrap();
    let entry = uri(&child.join("main.resin"));
    let library = temp.path().join("lib.resin");
    let library_uri = uri(&library);
    let source = "import { \"../lib.resin\" }; fn main() -> int  { answer() }";
    let mut client = Client::start(&child, Value::Null);
    client.open(&entry, source);
    client.diagnostics(&entry, Some(1), true);
    let library_text = "export { answer }; fn answer() -> int  { 7 }";
    std::fs::write(&library, library_text).unwrap();
    client.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": library_uri, "type": 1}]}),
    );
    client.diagnostics(&entry, Some(1), false);
    let changed = source.replace("../lib.resin", "./../lib.resin");
    client.change(&entry, 2, &changed);
    client.diagnostics(&entry, Some(2), false);
    let definition = client.request(
        "textDocument/definition",
        at(&entry, 0, changed.find("answer()").unwrap() as u32),
    );
    assert_eq!(definition["uri"], library_uri);
    std::fs::remove_file(&library).unwrap();
    client.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": library_uri, "type": 3}]}),
    );
    client.diagnostics(&entry, Some(2), true);
    std::fs::write(&library, library_text).unwrap();
    client.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes": [{"uri": library_uri, "type": 1}]}),
    );
    client.diagnostics(&entry, Some(2), false);
    let definition = client.request(
        "textDocument/definition",
        at(&entry, 0, changed.find("answer()").unwrap() as u32),
    );
    assert_eq!(definition["uri"], library_uri);
    client.stop();
}

#[test]
fn saved_editor_revision_is_a_cache_hit_for_an_independent_relocated_cli() {
    let editor_tree = TempDir::new().unwrap();
    let checkout = TempDir::new().unwrap();
    let main = "export { main }; import { \"helper.resin\" }; fn main() -> int  { answer() }";
    let old = "export { answer }; fn answer() -> int  { 3 }";
    let edited = "export { answer }; fn answer() -> int  { 7 }";
    std::fs::write(editor_tree.path().join("main.resin"), main).unwrap();
    std::fs::write(editor_tree.path().join("helper.resin"), old).unwrap();
    let mut client = Client::start(editor_tree.path(), Value::Null);
    let main_uri = uri(&editor_tree.path().join("main.resin"));
    let helper_uri = uri(&editor_tree.path().join("helper.resin"));
    client.open(&helper_uri, edited);
    client.open(&main_uri, main);
    client.diagnostics(&helper_uri, Some(1), false);
    client.diagnostics(&main_uri, Some(1), false);
    let before = client.service.server.counters();
    // Save exactly the editor bytes, then independently capture a relocated checkout.
    std::fs::write(editor_tree.path().join("helper.resin"), edited).unwrap();
    std::fs::write(checkout.path().join("main.resin"), main).unwrap();
    std::fs::write(checkout.path().join("helper.resin"), edited).unwrap();
    let output = client
        .service
        .command()
        .current_dir(checkout.path())
        .arg("main.resin")
        .arg("-o")
        .arg("program")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        Command::new(checkout.path().join("program"))
            .status()
            .unwrap()
            .code(),
        Some(7)
    );
    let shared = client.service.server.counters();
    assert_eq!(
        (
            shared.source_builds,
            shared.syntax_builds,
            shared.ast_builds,
            shared.hir_builds
        ),
        (
            before.source_builds,
            before.syntax_builds,
            before.ast_builds,
            before.hir_builds
        )
    );
    assert!(shared.verified_builds > before.verified_builds);
    // A different disk revision selects one new file and one new root analysis.
    std::fs::write(checkout.path().join("helper.resin"), old).unwrap();
    let output = client
        .service
        .command()
        .current_dir(checkout.path())
        .arg("main.resin")
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let different = client.service.server.counters();
    assert_eq!(different.source_builds, shared.source_builds + 1);
    assert_eq!(different.syntax_builds, shared.syntax_builds + 1);
    assert_eq!(different.ast_builds, shared.ast_builds + 1);
    assert_eq!(different.hir_builds, shared.hir_builds + 1);
    client.stop();
}

#[test]
fn managed_definitions_open_as_readonly_files_and_query_their_managed_identity() {
    let temp = TempDir::new().unwrap();
    let library = temp.path().join("library");
    let project = temp.path().join("project");
    std::fs::create_dir(&library).unwrap();
    std::fs::create_dir(&project).unwrap();
    let managed = "export { answer }; fn answer() -> int  { 42 }";
    std::fs::write(library.join("answer.resin"), managed).unwrap();
    let source = "import { \"$/answer.resin\" }; fn main() -> int  { answer() }";
    let mut client = Client::start_with_library_root(&project, Value::Null, Some(&library));
    let main_uri = uri(&project.join("main.resin"));
    client.open(&main_uri, source);
    client.diagnostics(&main_uri, Some(1), false);
    let definition = client.request(
        "textDocument/definition",
        at(&main_uri, 0, source.rfind("answer()").unwrap() as u32),
    );
    let managed_uri = definition["uri"]
        .as_str()
        .expect("managed definition file URI")
        .to_owned();
    let path = url::Url::parse(&managed_uri)
        .unwrap()
        .to_file_path()
        .unwrap();
    assert_ne!(path, library.join("answer.resin"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), managed);
    assert!(std::fs::metadata(&path).unwrap().permissions().readonly());
    client.open(&managed_uri, managed);
    client.diagnostics(&managed_uri, Some(1), false);
    let hover = client.request(
        "textDocument/hover",
        at(
            &managed_uri,
            0,
            managed.find("fn answer").unwrap() as u32 + 4,
        ),
    );
    assert!(
        hover["contents"]["value"]
            .as_str()
            .unwrap()
            .contains("answer")
    );
    client.stop();
}

#[test]
fn restarted_service_refreshes_managed_queries_without_an_editor_change() {
    let temp = TempDir::new().unwrap();
    let library = temp.path().join("library");
    let project = temp.path().join("project");
    std::fs::create_dir(&library).unwrap();
    std::fs::create_dir(&project).unwrap();
    let first = "export { answer }; fn answer() -> int  { 42 }";
    let second = "export { answer }; fn answer() -> bool  { 1 == 1 }";
    let managed_path = library.join("answer.resin");
    std::fs::write(&managed_path, first).unwrap();
    let source = "import { \"$/answer.resin\" }; fn main() -> int  { answer() }";
    let mut client = Client::start_with_library_root(&project, Value::Null, Some(&library));
    let main_uri = uri(&project.join("main.resin"));
    client.open(&main_uri, source);
    client.diagnostics(&main_uri, Some(1), false);
    let position = source.rfind("answer()").unwrap() as u32;
    let definition = client.request("textDocument/definition", at(&main_uri, 0, position));
    let old_uri = definition["uri"].as_str().unwrap().to_owned();
    let old_path = url::Url::parse(&old_uri).unwrap().to_file_path().unwrap();
    assert_eq!(std::fs::read_to_string(&old_path).unwrap(), first);

    std::fs::write(&managed_path, second).unwrap();
    client
        .service
        .restart(|config, _| config.library_root = library.clone());
    let hover = client.request("textDocument/hover", at(&main_uri, 0, position));
    assert!(hover.to_string().contains("bool"), "{hover}");
    client.diagnostics(&main_uri, Some(1), true);
    let definition = client.request("textDocument/definition", at(&main_uri, 0, position));
    let new_uri = definition["uri"].as_str().unwrap();
    assert_ne!(new_uri, old_uri);
    let new_path = url::Url::parse(new_uri).unwrap().to_file_path().unwrap();
    assert_eq!(std::fs::read_to_string(new_path).unwrap(), second);
    assert_eq!(std::fs::read_to_string(old_path).unwrap(), first);
    assert!(
        !project.join("main.resin").exists(),
        "unsaved editor source stays in memory"
    );
    client.stop();
}

#[test]
fn diagnostic_messages_are_single_line() {
    let temp = TempDir::new().unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let file = uri(&temp.path().join("diagnostic.resin"));
    client.open(&file, "fn main()  { let mut value: int = \"text\"; }");
    let diagnostics = client.diagnostics(&file, Some(1), true);
    for diagnostic in diagnostics.as_array().unwrap() {
        let message = diagnostic["message"].as_str().unwrap();
        assert!(!message.contains(['\n', '\r']), "{message}");
        assert!(!message.is_empty());
    }
    client.stop();
}

#[test]
fn doc_comment_hover_keeps_markdown_separate_from_the_signature_and_tracks_edits() {
    let temp = TempDir::new().unwrap();
    let mut client = Client::start(temp.path(), Value::Null);
    let uri = uri(&temp.path().join("docs.resin"));
    for (version, wording) in [(1, "Original"), (2, "Updated")] {
        let source = format!(
            "/// {wording} **Markdown**.\nfn read(value: int) -> int {{ value }}\nfn main() -> int {{ read(42) }}\n"
        );
        if version == 1 {
            client.open(&uri, &source);
        } else {
            client.change(&uri, version, &source);
        }
        client.diagnostics(&uri, Some(version), false);
        let hover = client.request("textDocument/hover", at(&uri, 2, 19));
        let markdown = hover["contents"]["value"].as_str().unwrap();
        assert!(
            markdown.contains(&format!("```\n\n{wording} **Markdown**.")),
            "{markdown}"
        );
    }
    client.stop();
}
