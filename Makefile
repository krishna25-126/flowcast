.PHONY: setup test check dashboard
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
test:
	python -m pytest -q
check:
	python scripts/check_data.py
dashboard:
	streamlit run dashboard/app.py
