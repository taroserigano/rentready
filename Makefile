# RentReady developer commands. Run `make help` for the list.
.PHONY: help neo4j backend frontend phoenix test eval report lint clean

VENV = .venv/bin

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

neo4j: ## Start the Neo4j graph database (Podman)
	podman run -d --name rentready-neo4j -p 7474:7474 -p 7687:7687 \
	  -e NEO4J_AUTH=neo4j/rentready123 \
	  -e 'NEO4J_PLUGINS=["apoc"]' \
	  -e 'NEO4J_dbms_security_procedures_unrestricted=apoc.*' \
	  docker.io/library/neo4j:5.26-community || podman start rentready-neo4j

backend: ## Run the FastAPI backend on :8000
	$(VENV)/uvicorn main:app --app-dir backend --port 8000 --reload

frontend: ## Run the React dev server on :5173
	cd frontend && npm run dev

phoenix: ## Run the Arize Phoenix trace UI on :6006
	PHOENIX_WORKING_DIR="$(PWD)/.phoenix" $(VENV)/python -m phoenix.server.main serve

test: ## Run backend + frontend test suites
	cd backend && ../$(VENV)/python -m pytest -q
	cd frontend && npm test

eval: ## Run the offline evaluation suite
	cd backend && EMBEDDING_BACKEND=hash ../$(VENV)/python -m evals.run_evals

report: ## Build the standalone HTML eval report (results/report.html)
	cd backend && ../$(VENV)/python -m evals.report

clean: ## Remove caches and local data stores
	rm -rf chroma_db rentready.db .phoenix backend/evals/results/*.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
