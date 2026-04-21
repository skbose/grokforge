# exp002 — GRPO on Apache Logs

**Goal:** First RL experiment scoped to Apache access log patterns.

**Setup:**
- Base: exp001 SFT checkpoint
- Reward: pygrok match (binary) + named capture coverage
- Scope: HTTPD_COMBINEDLOG and variants only
