default: dev

.PHONY: dev
dev: zero

.PHONY: clean
clean:
	rm -rf .venv

.PHONY: .venv
.venv: .venv/DONE
.venv/DONE:
	python3.14 -m venv .venv
	source .venv/bin/activate && pip install --upgrade pip poetry
	touch .venv/DONE

.PHONY: zero
zero: .venv
	source .venv/bin/activate && pip install -e ".[dev]"

.PHONY: demo
demo: .venv zero
	source .venv/bin/activate && python3 examples/demo.py

.PHONY: check
check: .venv zero
	.venv/bin/pyright .
