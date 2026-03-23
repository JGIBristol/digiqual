FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install system libraries needed for Python extensions (like SQLite for Positron)
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install 'just' command runner
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Install uv globally in the container
RUN pipx install uv

# Set up environment variables so the container knows where the venv will be
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
