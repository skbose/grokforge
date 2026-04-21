#!/usr/bin/env bash
# Download logstash-patterns-core into data/raw/
set -euo pipefail

mkdir -p data/raw
git clone --depth 1 https://github.com/logstash-plugins/logstash-patterns-core.git data/raw/logstash-patterns-core
