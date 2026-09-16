//! Verify a streamed executable before atomically publishing its local destination.
use crate::{BuiltArtifact, Error};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use resin_executor::{Cancellation, Execution};
use resin_protocol::{ArtifactKind, BuildMetadata, Target};
use std::path::Path;
use tokio::io::AsyncWriteExt;

pub(super) async fn receive(
    mut response: reqwest::Response,
    destination: &Path,
    revision: u64,
    target: &Target,
    instance: &str,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<BuiltArtifact, Error> {
    let metadata = metadata(&response, revision, target, instance)?;
    let destination = destination.to_owned();
    if destination.file_name().is_none() {
        return Err(Error::new("download destination must name a file"));
    }
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(Error::new)?;
    let directory = execution
        .run(cancellation, {
            let parent = parent.to_owned();
            move |_| tempfile::TempDir::new_in(parent)
        })
        .await
        .map_err(Error::new)?
        .map_err(Error::new)?;
    let temporary = directory.path().join("artifact");
    let mut output = tokio::fs::File::create(&temporary)
        .await
        .map_err(Error::new)?;
    let mut length = 0_u64;
    while let Some(chunk) = response.chunk().await.map_err(Error::new)? {
        cancellation.check().map_err(|_| Error::cancelled())?;
        length = length
            .checked_add(chunk.len() as u64)
            .ok_or_else(|| Error::new("artifact length overflow"))?;
        if length > metadata.artifact.length {
            return Err(Error::new("artifact exceeds its declared length"));
        }
        output.write_all(&chunk).await.map_err(Error::new)?;
    }
    output.flush().await.map_err(Error::new)?;
    drop(output);
    if length != metadata.artifact.length {
        return Err(Error::new("artifact does not match its declared length"));
    }
    let digest = execution
        .run(cancellation, {
            let temporary = temporary.clone();
            move |cancellation| digest(&temporary, cancellation)
        })
        .await
        .map_err(Error::new)??;
    if digest != metadata.artifact.blake3 {
        return Err(Error::new("artifact digest does not match its metadata"));
    }
    set_executable(&temporary).await?;
    cancellation.check().map_err(|_| Error::cancelled())?;
    tokio::fs::rename(&temporary, &destination)
        .await
        .map_err(Error::new)?;
    Ok(BuiltArtifact {
        path: destination,
        metadata,
    })
}

fn metadata(
    response: &reqwest::Response,
    revision: u64,
    target: &Target,
    instance: &str,
) -> Result<BuildMetadata, Error> {
    if response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        != Some("application/octet-stream")
    {
        return Err(Error::new(
            "build response is not an executable byte stream",
        ));
    }
    if response
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .is_some_and(|value| value != "identity")
    {
        return Err(Error::new("encoded artifact bodies are not supported"));
    }
    let header = response
        .headers()
        .get(resin_protocol::METADATA_HEADER)
        .ok_or_else(|| Error::new("build response has no artifact metadata"))?;
    if header.as_bytes().len() > resin_protocol::MAX_METADATA_BYTES {
        return Err(Error::new("artifact metadata exceeds the wire limit"));
    }
    let bytes = URL_SAFE_NO_PAD
        .decode(header.as_bytes())
        .map_err(|_| Error::new("artifact metadata is not base64url"))?;
    let metadata: BuildMetadata = serde_json::from_slice(&bytes).map_err(Error::new)?;
    validate_metadata(&metadata, revision, target, instance)?;
    let length = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    if length != Some(metadata.artifact.length) {
        return Err(Error::new(
            "HTTP artifact length does not match its metadata",
        ));
    }
    Ok(metadata)
}

fn validate_metadata(
    metadata: &BuildMetadata,
    revision: u64,
    target: &Target,
    instance: &str,
) -> Result<(), Error> {
    if metadata.revision != revision
        || &metadata.artifact.target != target
        || metadata.input.id.is_empty()
    {
        return Err(Error::new(
            "artifact metadata does not match the requested revision, server, or target",
        ));
    }
    if metadata.input.instance != instance {
        return Err(Error::instance_changed());
    }
    let artifact = &metadata.artifact;
    if artifact.kind != ArtifactKind::Executable || !artifact.executable {
        return Err(Error::new("server did not return a runnable executable"));
    }
    if !safe_filename(&artifact.name) {
        return Err(Error::new("artifact metadata contains an unsafe filename"));
    }
    if artifact.blake3.len() != 64
        || !artifact
            .blake3
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Error::new(
            "artifact metadata contains an invalid BLAKE3 digest",
        ));
    }
    Ok(())
}

fn safe_filename(name: &str) -> bool {
    let stem = name
        .split('.')
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    !name.is_empty()
        && name.len() <= 255
        && name != "."
        && name != ".."
        && !name.ends_with('.')
        && !matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        && !(stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
}

fn digest(path: &Path, cancellation: &Cancellation) -> Result<String, Error> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(Error::new)?;
    let mut hash = blake3::Hasher::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        cancellation.check().map_err(|_| Error::cancelled())?;
        let count = file.read(&mut buffer).map_err(Error::new)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hash.finalize().to_hex().to_string())
}

#[cfg(unix)]
async fn set_executable(path: &Path) -> Result<(), Error> {
    use std::os::unix::fs::PermissionsExt;
    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .await
        .map_err(Error::new)
}

#[cfg(not(unix))]
async fn set_executable(_: &Path) -> Result<(), Error> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(bytes: &[u8]) -> BuildMetadata {
        BuildMetadata {
            revision: 7,
            input: resin_protocol::InputHandle {
                instance: "instance".into(),
                id: "inputs".into(),
            },
            artifact: resin_protocol::ArtifactMetadata {
                name: "program".into(),
                kind: ArtifactKind::Executable,
                target: crate::host_target(),
                length: bytes.len() as u64,
                blake3: blake3::hash(bytes).to_hex().to_string(),
                executable: true,
            },
        }
    }

    async fn response(
        metadata: &BuildMetadata,
        bytes: Vec<u8>,
    ) -> (reqwest::Response, tokio::task::JoinHandle<()>) {
        let encoded = URL_SAFE_NO_PAD.encode(serde_json::to_vec(metadata).unwrap());
        let router = axum::Router::new().route(
            "/",
            axum::routing::get(move || {
                let bytes = bytes.clone();
                let encoded = encoded.clone();
                async move {
                    (
                        [
                            ("content-type", "application/octet-stream".to_owned()),
                            (resin_protocol::METADATA_HEADER, encoded),
                        ],
                        bytes,
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        (
            reqwest::get(format!("http://{address}")).await.unwrap(),
            server,
        )
    }

    #[tokio::test]
    async fn verified_bytes_replace_atomically_and_are_locally_executable() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("program");
        std::fs::write(&destination, b"previous").unwrap();
        let bytes = b"#!/bin/sh\nexit 23\n".to_vec();
        let metadata = artifact(&bytes);
        let (response, server) = response(&metadata, bytes.clone()).await;
        let built = receive(
            response,
            &destination,
            7,
            &crate::host_target(),
            "instance",
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(built.metadata(), &metadata);
        assert_eq!(tokio::fs::read(built.path()).await.unwrap(), bytes);
        #[cfg(unix)]
        assert_eq!(
            tokio::process::Command::new(built.path())
                .status()
                .await
                .unwrap()
                .code(),
            Some(23)
        );
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        server.abort();
    }

    #[tokio::test]
    async fn digest_and_metadata_failures_preserve_the_previous_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("program");
        std::fs::write(&destination, b"previous").unwrap();
        for issue in 0..5 {
            let bytes = b"replacement".to_vec();
            let mut metadata = artifact(&bytes);
            match issue {
                0 => metadata.artifact.blake3 = "0".repeat(64),
                1 => metadata.revision += 1,
                2 => metadata.artifact.target.architecture = "other".into(),
                3 => metadata.artifact.length += 1,
                _ => metadata.artifact.name = "../program".into(),
            }
            let (response, server) = response(&metadata, bytes).await;
            assert!(
                receive(
                    response,
                    &destination,
                    7,
                    &crate::host_target(),
                    "instance",
                    &Execution::default(),
                    &Cancellation::new()
                )
                .await
                .is_err()
            );
            assert_eq!(tokio::fs::read(&destination).await.unwrap(), b"previous");
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
            server.abort();
        }
    }

    #[tokio::test]
    async fn a_truncated_stream_removes_partial_bytes_and_preserves_the_previous_file() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("program");
        std::fs::write(&destination, b"previous").unwrap();
        let metadata = artifact(b"complete replacement");
        let encoded = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&metadata).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut connection, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1];
            connection.read_exact(&mut request).await.unwrap();
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nResin-Metadata: {encoded}\r\n\r\npartial",
                metadata.artifact.length
            );
            connection.write_all(header.as_bytes()).await.unwrap();
            connection.shutdown().await.unwrap();
        });
        let response = reqwest::get(format!("http://{address}")).await.unwrap();
        assert!(
            receive(
                response,
                &destination,
                7,
                &crate::host_target(),
                "instance",
                &Execution::default(),
                &Cancellation::new()
            )
            .await
            .is_err()
        );
        assert_eq!(std::fs::read(&destination).unwrap(), b"previous");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        server.await.unwrap();
    }

    #[test]
    fn artifact_names_never_identify_a_parent_or_nested_path() {
        for name in [
            "",
            ".",
            "..",
            "../program",
            "x/program",
            "C:program",
            "x\\program",
            "program\n",
            "program.",
            "CON.exe",
            "lpt1",
        ] {
            assert!(!safe_filename(name), "{name:?}");
        }
        assert!(safe_filename("program.exe"));
    }
}
