FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (debian)
RUN apt-get update \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 24 (required for codeql-development-mcp-server)
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install CodeQL CLI
RUN curl -Ls -o /tmp/codeql.zip https://github.com/github/codeql-cli-binaries/releases/latest/download/codeql-linux64.zip \
    && unzip /tmp/codeql.zip -d /opt \
    && mv /opt/codeql /opt/codeql-cli \
    && ln -s /opt/codeql-cli/codeql /usr/local/bin/codeql \
    && rm /tmp/codeql.zip

# Install seclab-taskflow-agent from PyPI
# For local testing with unpublished changes, replace with:
#   COPY . /app/seclab-taskflow-agent-src
#   RUN pip install /app/seclab-taskflow-agent-src
RUN pip install seclab-taskflow-agent

# Install CodeQL pack dependencies (legacy codeql toolbox)
RUN export SECLAB_TASKFLOW_AGENT=$(python -c 'import seclab_taskflow_agent as x; print(x.__path__[0])') && \
    codeql pack install $SECLAB_TASKFLOW_AGENT/mcp_servers/codeql/queries/mcp-cpp && \
    codeql pack install $SECLAB_TASKFLOW_AGENT/mcp_servers/codeql/queries/mcp-js

# Install codeql-development-mcp-server and its CodeQL tool query packs
RUN npm install -g codeql-development-mcp-server@2.25.2 \
    && codeql-development-mcp-server-setup-packs

# Default in-container paths consumed by the codeql_ql_mcp toolbox YAML
# (CODEQL_CLI / CODEQL_DBS_BASE_PATH / DATA_DIR / LOG_DIR). Mount host
# directories at these paths to override:
#   -v $HOME/codeql/databases:/codeql-databases  # MUST be writable:
#                                                # codeql_database_create writes
#                                                # new DBs into this directory
#   -v $HOME/seclab-data:/data
#   -v $HOME/seclab-logs:/logs
ENV CODEQL_CLI=/usr/local/bin/codeql \
    CODEQL_DBS_BASE_PATH=/codeql-databases \
    DATA_DIR=/data \
    LOG_DIR=/logs

RUN mkdir -p /codeql-databases /data /logs

VOLUME ["/codeql-databases", "/data", "/logs"]

ENTRYPOINT ["python", "-m", "seclab_taskflow_agent"]
