.PHONY: dev backend frontend

dev:
	@echo "Starting backend on :8000 and frontend on :3000..."
	@trap 'kill 0' INT; \
	uvicorn index:app --reload --host 127.0.0.1 --port 8000 & \
	cd frontend && npm run dev & \
	wait

backend:
	uvicorn index:app --reload --host 127.0.0.1 --port 8000

frontend:
	cd frontend && npm run dev
