@echo off
cd /d "C:\Users\dtmcg\RAG-LLM-project\financial-report-insights"
".venv\Scripts\python.exe" scripts\reembed_resumable.py >> ".cache\embeddings\_reembed_detached.log" 2>&1
