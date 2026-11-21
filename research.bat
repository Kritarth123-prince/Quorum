@echo off
REM Command-line research, e.g.:  research "Compare Qdrant and Milvus" --academic
REM Provider status / credits:     research --status
cd /d "%~dp0"
call agent\Scripts\activate.bat
python -m retrieval_pipeline.cli %*
