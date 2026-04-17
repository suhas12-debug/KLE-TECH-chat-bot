# Testing

This document describes the validation strategies and testing utilities used for the KLE Tech Chatbot.

## Manual Verification (UAT)
The primary method of verification is User Acceptance Testing (UAT) via the `chat.py` interface.
- **Goal**: Ensure the bot can distinguish between Semester 4 and Semester 6.
- **Test Scenarios**:
  - Greeting check ("Hi", "Hello").
  - Semantic similarity check ("Thursday timetable for 6th sem").
  - Conflict check (Ensure asking for 4th sem doesn't return 6th sem data).

## Automated Diagnostic Utilities

### `diagnostic_sim.py`
A specialist script used to calculate raw cosine similarity scores between specific queries and the entire vector database.
- **Function**: Prints top K results with their numerical scores.
- **Usage**: Use to debug why a specific query is triggering the fallback or retrieving the wrong context.

## Retrieval Tuning
- **Threshold**: 0.35. Scores below this are considered "uncertain" and routed to the office contact fallback.
- **Search depth**: Top 20 results are considered for filtering, ensuring a high recall rate before the Hybrid Filter is applied.
