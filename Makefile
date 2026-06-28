# Omaha-88 launcher. The app runs NATIVE (MLX needs the Apple GPU); only MinIO
# is dockerized. `make up` = start MinIO + the app in one go.

.PHONY: help setup minio minio-down run dev up down tunnel test qr

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install deps (incl. the vlm extra) and create .env if missing
	uv sync --extra vlm
	@[ -f .env ] || (cp .env.example .env && echo "created .env — edit OMAHA_ADMIN_TOKEN etc.")

minio: ## Start MinIO + auto-create the bucket (Docker, detached)
	docker compose up -d

minio-down: ## Stop MinIO (data persists)
	docker compose down

run: ## Run the app (native, MLX Moondream)
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

dev: ## Run the app with autoreload (no warmup cost worth it in dev)
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

up: run ## Start the app (the day-of "just go"). Uses R2 by default; for local
        ## MinIO instead, run `make minio` first.

down: minio-down ## Stop MinIO (only if you're using local MinIO)

tunnel: ## Run the Cloudflare tunnel (separate tab; needs cloudflared configured)
	cloudflared tunnel --config config/cloudflared.yml run

test: ## Run the test suite (no model weights needed)
	uv run pytest -q

qr: ## Generate the table-card QR code (SVG + PNG)
	uv run python scripts/generate_qr.py
