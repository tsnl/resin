use resin_common::prelude::*;
#[path = "support/pipeline.rs"]
mod pipeline;
#[path = "support/toolchain.rs"]
mod toolchain;
use std::{
    ffi::OsString,
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Output, Stdio},
};

struct Program {
    _temp: TempDir,
    executable: PathBuf,
}

impl Program {
    fn new(source: &str) -> Self {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        let path = temp.path().join("main.resin");
        fs::write(&path, source).unwrap();
        let module = pipeline::generate_program(&pipeline::load(&path).unwrap()).unwrap();
        Self::compile(temp, &resin_codegen::emit_c(&module, "main").unwrap())
    }

    fn compile(temp: TempDir, source: &str) -> Self {
        let executable = temp
            .path()
            .join(format!("program{}", std::env::consts::EXE_SUFFIX));
        let cc = std::env::var_os("CC")
            .unwrap_or_else(|| OsString::from(resin_platform_toolchain::DEFAULT_C_COMPILER));
        toolchain::c(&cc)
            .compile_c(source, &executable)
            .unwrap_or_else(|error| panic!("{error}\n{source}"));
        Self {
            _temp: temp,
            executable,
        }
    }

    fn run(&self, input: &[u8]) -> Output {
        let mut child = Command::new(&self.executable)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let input = input.to_vec();
        let writer = std::thread::spawn(move || stdin.write_all(&input).unwrap());
        let output = child.wait_with_output().unwrap();
        writer.join().unwrap();
        output
    }
}

#[test]
fn lines_preserve_bytes_and_distinguish_empty_lines_from_eof() {
    let program = Program::new(
        r#"
        export { main };
        import { "std/console.resin" };
        def failed(error: InputError) -> Result<(), InputError> = { err(error) };
        def main() -> Result<(), _> = {
            var reading = 1 == 1;
            while (reading) {
                match (Console.read_line()) {
                    ok(line) => {

                        if (Ptr<ubyte>(ulong(line.data) + line.length).* != ubyte(0)) {
                            print("missing terminator");
                        } else {};
                        print(fmt("[{0}:", (line.length,)));
                        Console.print(line)?;
                        print("]");
                    },
                    err(error) => {
                        match (error) {
                            EndOfInput(e) => { reading := 1 == 0; },
                            InputReadError(e) => { failed(e)?; },
                            InputOutOfMemory(e) => { failed(e)?; },
                        };
                    },
                };
            };
            ok(())
        };
    "#,
    );
    for (input, expected) in [
        (&b""[..], &b""[..]),
        (b"\n", b"[0:]"),
        (b"\r\n", b"[0:]"),
        (b"one\ntwo\r\n\nlast", b"[3:one][3:two][0:][4:last]"),
        (b"a\rb\r", b"[4:a\rb\r]"),
        (b"a\0b\0\n", b"[4:a\0b\0]"),
        ("héllo 世界\n".as_bytes(), "[13:héllo 世界]".as_bytes()),
        (b"\xff\x80\n", b"[2:\xff\x80]"),
    ] {
        let output = program.run(input);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected);
        assert!(output.stderr.is_empty());
    }
    for length in [126, 127, 128, 255, 256, 100_000] {
        let mut input = vec![b'x'; length];
        input.extend_from_slice(b"\r\n");
        let mut expected = format!("[{length}:").into_bytes();
        expected.extend(std::iter::repeat_n(b'x', length));
        expected.push(b']');
        let output = program.run(&input);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected);
    }
}

#[test]
fn byte_input_distinguishes_bytes_from_eof() {
    let program = Program::new(
        r#"
        export { main };
        import { "std/console.resin" };
        def main() -> Result<int, _> = {
            var zero = Console.read_byte()?;
            var first = Console.read_byte()?;
            var ended = match (Console.read_byte()) {
                ok(byte) => { 1 == 0 },
                err(error) => {
                    match (error) {
                        EndOfInput(e) => { 1 == 1 },
                        InputReadError(e) => { 1 == 0 },
                    }
                },
            };
            ok(if (zero == ubyte(0) && first == ubyte(255) && ended) { 0 } else { 1 })
        };
    "#,
    );
    assert!(program.run(b"\0\xff").status.success());
    let output = program.run(b"");
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unhandled error: EndOfInput"));
}

#[test]
fn greeting_example_and_eof_error() {
    let program = Program::new(include_str!("../examples/input.resin"));
    let output = program.run(b"Ada\n");
    assert!(output.status.success());
    assert_eq!(output.stdout, b"What is your name? Hello, Ada!\n");
    let output = program.run(b"");
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("EndOfInput"));
}

#[test]
fn failures_release_the_current_buffer_and_report_the_right_error() {
    for mode in 0..=6 {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        let header = temp.path().join("faults.h");
        fs::write(
            &header,
            format!(
                "#define TEST_MODE {mode}\n{}",
                include_str!("support/console_faults.h")
            ),
        )
        .unwrap();
        let header = header.to_string_lossy().replace('\\', "/");
        let source = format!(
            r#"
            export {{ main }};
            import {{ "std/console.resin" }};
            extern "{header}" def console_test_mode() -> int;
            extern "{header}" def console_test_frees() -> int;
            def exercise() -> int = {{
                var mode = console_test_mode();
                match (Console.read_line()) {{
                    ok(line) => {{

                        match (Console.print(line)) {{
                            ok(unit) => {{ if (mode == 6 && line.length == ulong(300)) {{ 0 }} else {{ 1 }} }},
                            err(error) => {{ if (mode == 4 || mode == 5) {{ 0 }} else {{ 2 }} }},
                        }}
                    }},
                    err(error) => {{
                        match (error) {{
                            EndOfInput(e) => {{ 3 }},
                            InputReadError(e) => {{ if (mode == 2 || mode == 3) {{ 0 }} else {{ 4 }} }},
                            InputOutOfMemory(e) => {{ if (mode == 0 || mode == 1) {{ 0 }} else {{ 5 }} }},
                        }}
                    }},
                }}
            }};
            def main() -> int = {{
                var result = exercise();
                var expected = if (console_test_mode() == 0) {{ 0 }} else {{ 1 }};
                if (console_test_frees() != expected) {{ 6 }} else {{ result }}
            }};
        "#
        );
        let path = temp.path().join("main.resin");
        fs::write(&path, source).unwrap();
        let module = pipeline::generate_program(&pipeline::load(&path).unwrap()).unwrap();
        // Include before the generated header list so all foreign calls use the test shims.
        let source = format!(
            "#include \"{header}\"\n{}",
            resin_codegen::emit_c(&module, "main").unwrap()
        );
        let output = Program::compile(temp, &source).run(b"");
        assert!(
            output.status.success(),
            "mode {mode}: {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn streams_write_literals_and_owned_strings_verbatim() {
    let program = Program::new(
        r#"
        export { main };
        import { "std/io.resin" };
        def literal() -> Span<ubyte> = { "static\0bytes" };
        def main() -> Result<(), _> = {
            var out = Io.stdout();
            var error = Io.stderr();
            var text = fmt("n = {0}", (42,));
            var copy = text;
            out.write("raw {0}\0")?;
            out.write(copy)?;
            error.write(fmt("error: {0}\n", (text,)))?;
            error.write(literal())?;
            ok(())
        };
    "#,
    );
    let output = program.run(b"");
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(output.stdout, b"raw {0}\0n = 42");
    assert_eq!(output.stderr, b"error: n = 42\nstatic\0bytes");
}

#[test]
fn stream_write_failure_propagates_as_a_library_error() {
    let program = Program::new(
        r#"
        export { main };
        import { "std/io.resin" };
        def main() -> Result<(), _> = {
            Output { stream = 99_ui }.write("unwritten")?;
            print("not reached");
            ok(())
        };
    "#,
    );
    let output = program.run(b"");
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("WriteError"));
}
