.PHONY: check format

check:
	cargo test --workspace
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

format:
	cargo fmt --all
