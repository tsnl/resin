use crate::Error;
use std::process::Command;

pub(super) fn run(mut command: Command, kind: &str) -> Result<(), Error> {
    let result = command.output().map_err(|error| {
        Error(format!(
            "cannot run {}: {error}",
            command.get_program().to_string_lossy()
        ))
    })?;
    if result.status.success() {
        return Ok(());
    }
    Err(Error(format!(
        "{kind} failed ({}):\n{}{}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    )))
}
