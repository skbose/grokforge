# exp001 — SFT Baseline

**Goal:** Establish a supervised fine-tuning baseline before adding RL.

**Setup:**
- Model: Qwen2.5-Coder-1.5B-Instruct
- Data: synthetic (log_line → grok_pattern) pairs from logstash-patterns-core
- Metric: exact match + pygrok match rate on eval set
