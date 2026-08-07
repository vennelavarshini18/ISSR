# Phase 4 Roadmap: Communication Analytics

This document outlines the final phase of the TCAMP project (August). 
The goal of Phase 4 is to take our final audio transcripts and turn them into mathematical measurements that Human Factors researchers can actually use to study team behavior.

## Part 1: Behavioral Metrics Extraction
The analytics module (`tcamp/analytics/behavioral_metrics.py`) calculates 6 core communication metrics from the transcript:

1. **Total Talk Time:** How long each person spoke (measures dominance).
2. **Silence Ratio:** The percentage of the meeting where no one was talking.
3. **Turn-Taking Rate:** How often the speaker changes.
4. **Interruptions (Overlaps):** How many times speakers talked over each other.
5. **Response Latency:** The delay (in seconds) before someone responds to another person.
6. **Centralization (Gini Coefficient):** A score showing if participation was equal or one-sided.

## Part 2: Local AI Text Tagging (Ollama)
A local LLM integration (Llama 3.2 via Ollama) provides semantic analysis:
* The model analyzes each utterance and tags it with a semantic label (e.g., "Instruction", "Question", "Acknowledgment", "Warning").
* This executes locally to ensure clinical data privacy.

## Part 3: Data Export and Visualization
* **Export:** All metrics and AI tags are saved to a structured JSON format for researchers to load into Excel or SPSS.
* **Dashboard:** A visualization module utilizing `plotly` to chart team communication dynamics.
