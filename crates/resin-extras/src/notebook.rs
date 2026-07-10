//! Streaming self-contained HTML notebooks.
//!
//! Callers append sections of text, images, and interactive Plotly figures to a single
//! HTML file.
//!
//! ## Viewing with live-server
//!
//! Use [`live-server`](https://crates.io/crates/live-server) so the browser
//! reloads only when the file changes:
//!
//! ```text
//! cargo install live-server
//! live-server --path . --open out/my-notebook.html
//! ```

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use plotly::Plot;

/// A growing HTML report. Create once, then append sections as the run proceeds.
pub struct Notebook {
    path: PathBuf,
    plot_n: usize,
}

impl Notebook {
    /// Truncate `path` and write the HTML skeleton (inlined Plotly.js + open body).
    ///
    /// `title` becomes `<title>…</title>` (HTML-escaped). Closing
    /// `</body></html>` is omitted so partial files still preview while the
    /// program streams more content.
    pub fn create(path: impl AsRef<Path>, title: &str) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        let mut f = File::create(&path).map_err(|e| e.to_string())?;
        write_head(&mut f, title)?;
        f.write_all(Plot::offline_js_sources().as_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(HEAD_CLOSE_BODY_OPEN.as_bytes())
            .map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
        Ok(Self { path, plot_n: 0 })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn h1(&mut self, text: &str) -> Result<(), String> {
        self.append(&format!("<h1>{}</h1>\n", escape(text)))
    }

    pub fn h2(&mut self, text: &str) -> Result<(), String> {
        self.append(&format!("<h2>{}</h2>\n", escape(text)))
    }

    pub fn p(&mut self, text: &str) -> Result<(), String> {
        self.append(&format!("<p>{}</p>\n", escape(text)))
    }

    /// Append raw HTML (caller is responsible for safety and validity).
    pub fn html(&mut self, raw: &str) -> Result<(), String> {
        self.append(raw)
    }

    /// Embed an interactive Plotly figure (requires the inlined Plotly.js head).
    pub fn plot(&mut self, plot: &Plot) -> Result<(), String> {
        self.plot_n += 1;
        let id = format!("plot-{}", self.plot_n);
        let inline = plot.to_inline_html(Some(&id));
        self.html(&format!("<div class=\"plot\">{inline}</div>"))
    }

    /// Embed a PNG as a self-contained `data:` image (no external files).
    pub fn image_png(&mut self, alt: &str, png: &[u8]) -> Result<(), String> {
        let b64 = base64_encode(png);
        self.html(&format!(
            "<p><center><img alt=\"{}\" src=\"data:image/png;base64,{b64}\"/></center></p>",
            escape(alt)
        ))
    }

    /// Append a horizontal rule `<hr/>`.
    pub fn hr(&mut self) -> Result<(), String> {
        self.html("<hr/>")
    }

    fn append(&mut self, chunk: &str) -> Result<(), String> {
        let mut f = OpenOptions::new()
            .append(true)
            .open(&self.path)
            .map_err(|e| e.to_string())?;
        f.write_all(chunk.as_bytes()).map_err(|e| e.to_string())?;
        f.write(b"\n\n").map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
        Ok(())
    }
}

const HEAD_CLOSE_BODY_OPEN: &str = r#"
</head>
<body>
"#;

fn write_head(f: &mut File, title: &str) -> Result<(), String> {
    writeln!(f, "<!DOCTYPE html>").map_err(|e| e.to_string())?;
    writeln!(f, "<html lang=\"en\">").map_err(|e| e.to_string())?;
    writeln!(f, "<head>").map_err(|e| e.to_string())?;
    writeln!(f, "<meta charset=\"utf-8\"/>").map_err(|e| e.to_string())?;
    writeln!(
        f,
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
    )
    .map_err(|e| e.to_string())?;
    writeln!(f, "<title>{}</title>", escape(title)).map_err(|e| e.to_string())?;
    writeln!(
        f,
        r#"<style>
  body {{ font-family: system-ui, sans-serif; max-width: 52rem; margin: 1.5rem auto; padding: 0 1rem; line-height: 1.5; color: #111; }}
  h1, h2 {{ line-height: 1.2; }}
  .plot {{ margin: 1rem 0; }}
  img {{ max-width: 100%; height: auto; }}
  code, pre {{ font-family: ui-monospace, monospace; }}
  pre {{ background: #f4f4f5; padding: 0.75rem 1rem; overflow-x: auto; }}
</style>"#
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= data.len() {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8) | (data[i + 2] as u32);
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(T[((n >> 6) & 63) as usize] as char);
        out.push(T[(n & 63) as usize] as char);
        i += 3;
    }
    match data.len() - i {
        1 => {
            let n = (data[i] as u32) << 16;
            out.push(T[((n >> 18) & 63) as usize] as char);
            out.push(T[((n >> 12) & 63) as usize] as char);
            out.push_str("==");
        }
        2 => {
            let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8);
            out.push(T[((n >> 18) & 63) as usize] as char);
            out.push(T[((n >> 12) & 63) as usize] as char);
            out.push(T[((n >> 6) & 63) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use plotly::common::Mode;
    use plotly::layout::Layout;
    use plotly::{Plot, Scatter};
    use std::fs;

    #[test]
    fn notebook_streams_self_contained_html() {
        let dir = std::env::temp_dir().join("resin-extras-notebook-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("note.html");

        let mut nb = Notebook::create(&path, "k-NN notes").unwrap();
        nb.h1("Hello").unwrap();
        nb.p("streaming").unwrap();
        nb.image_png("dot", &MINI_PNG).unwrap();

        let mut plot = Plot::new();
        plot.add_trace(
            Scatter::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]).mode(Mode::LinesMarkers),
        );
        plot.set_layout(Layout::new().title("line"));
        nb.plot(&plot).unwrap();

        let html = fs::read_to_string(&path).unwrap();
        assert!(html.contains("<title>k-NN notes</title>"));
        assert!(!html.contains("http-equiv=\"refresh\""));
        assert!(html.contains("<h1>Hello</h1>"));
        assert!(html.contains("data:image/png;base64,"));
        assert!(html.contains("Plotly.newPlot") || html.contains("Plotly."));
        // Offline Plotly.js is large and inlined in the head.
        assert!(html.contains("plotly") || html.len() > 100_000);
        assert!(!html.contains("</html>"));
    }

    /// 1×1 transparent PNG.
    const MINI_PNG: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
        0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];
}
