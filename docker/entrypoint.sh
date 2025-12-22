#!/bin/bash
# Docker entrypoint script for memory-benchmark-harness
#
# This script initializes a git repository if needed (for git-notes adapter)
# and then passes through to the benchmark CLI.

set -e

# Initialize git repo if not present (required for git-notes adapter)
if [ ! -d "/app/.git" ]; then
    echo "Initializing git repository for git-notes adapter..."
    git init -b main /app
    git -C /app config user.email "benchmark@container.local"
    git -C /app config user.name "Benchmark Harness"

    # Create initial commit with existing files
    git -C /app add -A
    git -C /app commit -m "Initial commit for benchmark" --allow-empty --quiet
    echo "Git repository initialized."
fi

# Execute the command passed to the container
exec python -m src.cli.main "$@"
