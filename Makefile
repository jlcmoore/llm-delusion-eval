# Makefile

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: clean pyfmt ruff pylint mdfmt jsfmt jslint

clean:
	rm -rf .ruff_cache __pycache__ .venv

# Code formatting and fast linting: ruff
pyfmt: ruff
ruff:
	uv run ruff check --fix --unsafe-fixes src
	uv run ruff format src

# Static analysis: pylint (heavy linting)
pylint:
	@pylint_status=0; \
	uv run pylint --output-format=colorized src || pylint_status=$$?; \
	uv run vulture src || true; \
	exit $$pylint_status

mdfmt:
	npx prettier --write '**/*.md'

jsfmt:
	npx prettier --write 'src/llm_delusion_eval/scripts/report_assets/**/*.{js,css,html}'

jslint:
	npx prettier --check 'src/llm_delusion_eval/scripts/report_assets/**/*.{js,css,html}'
	npx eslint --cache --fix "src/llm_delusion_eval/scripts/report_assets/**/*.js"
	npx jscpd --gitignore --pattern 'src/llm_delusion_eval/scripts/report_assets/**/*.{js,mjs,cjs}' --reporters consoleFull
