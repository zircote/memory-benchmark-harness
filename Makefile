# Memory Benchmark Harness - Makefile
# Primary interface for development commands
# Requires: uv (preferred) or pip

.PHONY: all install dev test test-unit test-integration test-e2e test-all lint format typecheck clean check help pre-commit coverage docs docker-build-cpu docker-build-gpu docker-test docker-test-e2e docker-test-all docker-smoke docker-benchmark-locomo docker-benchmark-lme docker-benchmark-contextbench docker-benchmark-mab docker-benchmark-all docker-benchmark-parallel docker-benchmark-quick benchmark-report benchmark-publication benchmark-full-pipeline

# Default Python version for local development
PYTHON_VERSION ?= 3.11

# Detect package manager (prefer uv)
UV := $(shell command -v uv 2> /dev/null)
ifdef UV
	PKG_MGR := uv
	RUN := uv run
	INSTALL := uv sync
	INSTALL_DEV := uv sync --all-groups
	INSTALL_PUB := uv pip install -e ".[publication]"
	PYTHON := uv run python
else
	PKG_MGR := pip
	RUN :=
	INSTALL := pip install -e .
	INSTALL_DEV := pip install -e ".[dev]"
	INSTALL_PUB := pip install -e ".[publication]"
	PYTHON := python
endif

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

all: lint typecheck test ## Run all quality checks (lint, typecheck, test)

##@ Setup

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies with $(PKG_MGR)...$(NC)"
ifdef UV
	$(INSTALL)
else
	$(INSTALL)
endif

dev: ## Install all dependencies including dev tools
	@echo "$(BLUE)Installing dev dependencies with $(PKG_MGR)...$(NC)"
ifdef UV
	$(INSTALL_DEV)
else
	$(INSTALL_DEV)
endif
	@echo "$(GREEN)Setting up pre-commit hooks...$(NC)"
	$(RUN) pre-commit install

pre-commit: ## Install/update pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	$(RUN) pre-commit install
	$(RUN) pre-commit autoupdate

##@ Quality Checks

lint: ## Run ruff linter
	@echo "$(BLUE)Running ruff linter...$(NC)"
	$(RUN) ruff check src tests scripts

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code with ruff...$(NC)"
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(RUN) ruff format --check src tests scripts

typecheck: ## Run mypy type checker
	@echo "$(BLUE)Running mypy type checker...$(NC)"
	$(RUN) mypy src --ignore-missing-imports

check: lint format-check typecheck ## Run all checks without tests (CI fast path)
	@echo "$(GREEN)All quality checks passed!$(NC)"

##@ Testing

test: ## Run all tests with coverage
	@echo "$(BLUE)Running all tests...$(NC)"
	$(RUN) pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(RUN) pytest tests/unit -v --tb=short --cov=src --cov-report=term-missing

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(RUN) pytest tests/integration -v --tb=short

test-quick: ## Run tests without coverage (faster)
	@echo "$(BLUE)Running tests (no coverage)...$(NC)"
	$(RUN) pytest tests/unit -v --tb=short

test-e2e: ## Run e2e tests (requires network for dataset downloads)
	@echo "$(BLUE)Running e2e tests...$(NC)"
	$(RUN) pytest tests/ -v --tb=short -m e2e

test-all: ## Run all tests including e2e
	@echo "$(BLUE)Running all tests...$(NC)"
	$(RUN) pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

coverage: ## Generate HTML coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	$(RUN) pytest tests/ --cov=src --cov-report=html --cov-report=xml
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(NC)"

##@ Development

run: ## Run the benchmark CLI
	@echo "$(BLUE)Running benchmark CLI...$(NC)"
	$(RUN) benchmark $(ARGS)

shell: ## Open a Python shell with project imports
	@echo "$(BLUE)Opening Python shell...$(NC)"
	$(RUN) python -i -c "from src import *; print('Memory Benchmark Harness loaded')"

##@ Maintenance

clean: ## Remove build artifacts and caches
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean ## Remove all generated files including uv cache
	@echo "$(YELLOW)Deep cleaning...$(NC)"
	rm -rf .venv/
ifdef UV
	uv cache clean
endif
	@echo "$(GREEN)Deep clean complete!$(NC)"

##@ CI Helpers

quality: ci-lint ci-typecheck ci-test ## Run all quality gates (mirrors GitHub Actions CI exactly)
	@echo "$(GREEN)All quality gates passed!$(NC)"

ci: quality ## Alias for quality (run all CI checks locally)

ci-lint: ## CI: Lint + format check (exact CI parity)
	@echo "$(BLUE)Running lint checks (CI parity)...$(NC)"
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

ci-typecheck: ## CI: Type checking (exact CI parity)
	@echo "$(BLUE)Running type checks (CI parity)...$(NC)"
	$(RUN) mypy src --ignore-missing-imports

ci-test: ## CI: Unit tests with coverage (exact CI parity)
	@echo "$(BLUE)Running unit tests (CI parity)...$(NC)"
	$(RUN) pytest tests/unit -v --tb=short --cov=src --cov-report=xml --cov-report=term-missing

##@ Docker

docker-build-cpu: ## Build CPU Docker image
	docker build -t benchmark-harness:cpu -f docker/Dockerfile.cpu .

docker-build-gpu: ## Build GPU Docker image
	docker build -t benchmark-harness:gpu -f docker/Dockerfile.gpu .

docker-test: docker-build-cpu ## Run unit tests in Docker (excludes e2e)
	@echo "$(BLUE)Running unit tests in Docker...$(NC)"
	docker compose -f docker/docker-compose.yml run --rm test

docker-test-e2e: docker-build-cpu ## Run e2e tests in Docker (downloads datasets)
	@echo "$(BLUE)Running e2e tests in Docker...$(NC)"
	docker compose -f docker/docker-compose.yml run --rm test-e2e

docker-test-all: docker-build-cpu ## Run all tests in Docker (unit + e2e)
	@echo "$(BLUE)Running all tests in Docker...$(NC)"
	docker compose -f docker/docker-compose.yml run --rm test
	docker compose -f docker/docker-compose.yml run --rm test-e2e

docker-smoke: docker-build-cpu ## Quick Docker sanity check (import test)
	docker run --rm benchmark-harness:cpu python -c "import src.adapters; print('Import successful')"

##@ Benchmark Experiments (Publication)

# Configuration for publication runs
BENCHMARK_TRIALS ?= 5
BENCHMARK_ADAPTERS ?= git-notes,no-memory
BENCHMARK_OUTPUT ?= results

docker-benchmark-locomo: docker-build-cpu ## Run LoCoMo benchmark in Docker
	@echo "$(BLUE)Running LoCoMo benchmark (trials=$(BENCHMARK_TRIALS), adapters=$(BENCHMARK_ADAPTERS))...$(NC)"
	@mkdir -p $(BENCHMARK_OUTPUT)
	docker compose -f docker/docker-compose.yml run --rm \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		benchmark-lite run locomo \
		--adapter $(BENCHMARK_ADAPTERS) \
		--trials $(BENCHMARK_TRIALS) \
		--output /app/results
	@echo "$(GREEN)LoCoMo results saved to $(BENCHMARK_OUTPUT)/$(NC)"

docker-benchmark-lme: docker-build-cpu ## Run LongMemEval benchmark in Docker
	@echo "$(BLUE)Running LongMemEval benchmark (trials=$(BENCHMARK_TRIALS), adapters=$(BENCHMARK_ADAPTERS))...$(NC)"
	@mkdir -p $(BENCHMARK_OUTPUT)
	docker compose -f docker/docker-compose.yml run --rm \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		benchmark-lite run longmemeval \
		--adapter $(BENCHMARK_ADAPTERS) \
		--trials $(BENCHMARK_TRIALS) \
		--output /app/results
	@echo "$(GREEN)LongMemEval results saved to $(BENCHMARK_OUTPUT)/$(NC)"

docker-benchmark-contextbench: docker-build-cpu ## Run Context-Bench benchmark in Docker
	@echo "$(BLUE)Running Context-Bench benchmark (trials=$(BENCHMARK_TRIALS), adapters=$(BENCHMARK_ADAPTERS))...$(NC)"
	@mkdir -p $(BENCHMARK_OUTPUT)
	docker compose -f docker/docker-compose.yml run --rm \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		benchmark-lite run contextbench \
		--adapter $(BENCHMARK_ADAPTERS) \
		--trials $(BENCHMARK_TRIALS) \
		--output /app/results
	@echo "$(GREEN)Context-Bench results saved to $(BENCHMARK_OUTPUT)/$(NC)"

docker-benchmark-mab: docker-build-cpu ## Run MemoryAgentBench benchmark in Docker
	@echo "$(BLUE)Running MemoryAgentBench benchmark (trials=$(BENCHMARK_TRIALS), adapters=$(BENCHMARK_ADAPTERS))...$(NC)"
	@mkdir -p $(BENCHMARK_OUTPUT)
	docker compose -f docker/docker-compose.yml run --rm \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		benchmark-lite run memoryagentbench \
		--adapter $(BENCHMARK_ADAPTERS) \
		--trials $(BENCHMARK_TRIALS) \
		--output /app/results
	@echo "$(GREEN)MemoryAgentBench results saved to $(BENCHMARK_OUTPUT)/$(NC)"

docker-benchmark-all: docker-build-cpu ## Run ALL benchmarks in Docker (sequential)
	@echo "$(BLUE)Running all benchmarks for publication (sequential)...$(NC)"
	@echo "$(YELLOW)Configuration: trials=$(BENCHMARK_TRIALS), adapters=$(BENCHMARK_ADAPTERS)$(NC)"
	@mkdir -p $(BENCHMARK_OUTPUT)
	$(MAKE) docker-benchmark-locomo
	$(MAKE) docker-benchmark-lme
	$(MAKE) docker-benchmark-contextbench
	$(MAKE) docker-benchmark-mab
	@echo "$(GREEN)All benchmark results saved to $(BENCHMARK_OUTPUT)/$(NC)"

docker-benchmark-parallel: docker-build-cpu ## Run ALL benchmarks in parallel (faster, needs resources)
	@echo "$(BLUE)Running all benchmarks in parallel...$(NC)"
	@echo "$(YELLOW)Configuration: trials=$(BENCHMARK_TRIALS), adapters=$(BENCHMARK_ADAPTERS)$(NC)"
	@echo "$(YELLOW)Note: Requires sufficient CPU/memory and API rate limits$(NC)"
	@mkdir -p $(BENCHMARK_OUTPUT)
	@docker compose -f docker/docker-compose.yml run --rm -e OPENAI_API_KEY=$(OPENAI_API_KEY) benchmark-lite run locomo --adapter $(BENCHMARK_ADAPTERS) --trials $(BENCHMARK_TRIALS) --output /app/results & \
	docker compose -f docker/docker-compose.yml run --rm -e OPENAI_API_KEY=$(OPENAI_API_KEY) benchmark-lite run longmemeval --adapter $(BENCHMARK_ADAPTERS) --trials $(BENCHMARK_TRIALS) --output /app/results & \
	docker compose -f docker/docker-compose.yml run --rm -e OPENAI_API_KEY=$(OPENAI_API_KEY) benchmark-lite run contextbench --adapter $(BENCHMARK_ADAPTERS) --trials $(BENCHMARK_TRIALS) --output /app/results & \
	docker compose -f docker/docker-compose.yml run --rm -e OPENAI_API_KEY=$(OPENAI_API_KEY) benchmark-lite run memoryagentbench --adapter $(BENCHMARK_ADAPTERS) --trials $(BENCHMARK_TRIALS) --output /app/results & \
	wait
	@echo "$(GREEN)All benchmark results saved to $(BENCHMARK_OUTPUT)/$(NC)"

docker-benchmark-quick: docker-build-cpu ## Quick benchmark run (1 trial, mock adapter)
	@echo "$(BLUE)Running quick benchmark validation...$(NC)"
	@mkdir -p $(BENCHMARK_OUTPUT)
	docker compose -f docker/docker-compose.yml run --rm \
		benchmark-lite run locomo \
		--adapter mock \
		--trials 1 \
		--output /app/results
	@echo "$(GREEN)Quick validation complete$(NC)"

benchmark-report: ## Generate reports from benchmark results
	@echo "$(BLUE)Generating benchmark reports...$(NC)"
	@if ls $(BENCHMARK_OUTPUT)/exp_*.json 1>/dev/null 2>&1; then \
		for f in $(BENCHMARK_OUTPUT)/exp_*.json; do \
			$(RUN) benchmark report "$$f" --output $(BENCHMARK_OUTPUT)/reports/; \
		done; \
		echo "$(GREEN)Reports saved to $(BENCHMARK_OUTPUT)/reports/$(NC)"; \
	else \
		echo "$(RED)No experiment results found in $(BENCHMARK_OUTPUT)/$(NC)"; \
		exit 1; \
	fi

benchmark-publication: ## Generate publication artifacts (tables, figures, stats)
	@echo "$(BLUE)Ensuring publication dependencies are installed...$(NC)"
	@$(INSTALL_PUB) > /dev/null 2>&1 || $(INSTALL_PUB)
	@echo "$(BLUE)Generating publication artifacts...$(NC)"
	$(RUN) benchmark publication all $(BENCHMARK_OUTPUT)/ --output $(BENCHMARK_OUTPUT)/publication/
	@echo "$(GREEN)Publication artifacts saved to $(BENCHMARK_OUTPUT)/publication/$(NC)"

benchmark-full-pipeline: docker-benchmark-all benchmark-report benchmark-publication ## Full pipeline: run benchmarks + generate all artifacts
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Full benchmark pipeline complete!$(NC)"
	@echo "$(GREEN)Results: $(BENCHMARK_OUTPUT)/$(NC)"
	@echo "$(GREEN)Reports: $(BENCHMARK_OUTPUT)/reports/$(NC)"
	@echo "$(GREEN)Publication: $(BENCHMARK_OUTPUT)/publication/$(NC)"
	@echo "$(GREEN)========================================$(NC)"
