# Conventions

This document outlines the coding standards, patterns, and best practices observed in this project.

## Modeling Conventions
- **Quantization**: Always use 4-bit NF4 (`bitsandbytes`) for the LLM to minimize VRAM footprint (~400MB total).
- **Compute Precision**: Use `bfloat16` for computation to maintain numerical stability in the LLM.
- **Resource Allocation**:
  - **SBERT**: Runs on **CPU** to avoid conflict with LLM VRAM.
  - **LLM**: Runs on **GPU** for fast token generation.

## Data Normalization
- **Semester Identifiers**: Avoid Roman numerals (IV, VI) in strings. Use word-form `Semester 4` or `Semester 6` to improve semantic retrieval accuracy.
- **Fact Prefixing**: Every retrieved fact is prefixed with `[KLE Tech University Knowledge]` to ground the model in the specific domain.

## UI/CLI Patterns
- **User Feedback**: Use ANSI Color codes (Cyan for User, Green for Bot, Yellow for Status) to maintain readable CLI logs.
- **Graceful Failures**: If search similarity is below 0.35, the system must trigger a standard "Contact university office" fallback rather than guessing.

## Error Handling
- **Missing Dependencies**: Check for `.npy` and `.json` files at startup and exit early with a clear error message if absent.
