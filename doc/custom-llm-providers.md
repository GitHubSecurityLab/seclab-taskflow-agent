# Custom LLM Providers Configuration

This guide explains how to configure the Taskflow Agent to use custom OpenAI-compatible endpoints, including OpenRouter, vLLM, Ollama, and other providers.

## Overview

The Taskflow Agent supports any OpenAI-compatible API endpoint through the `openai_agents` backend. You can configure custom endpoints either globally via environment variables or per-model in a `model_config` YAML file.

## Configuration Methods

### Method 1: Environment Variables (Global)

Set these environment variables to configure a global custom endpoint:

```bash
# Required: Your custom endpoint URL
export AI_API_ENDPOINT="https://your-provider.com/v1"

# Required: API key for authentication
export AI_API_TOKEN="your-api-key-here"
```

This configuration applies to all models unless overridden in a `model_config` file.

### Method 2: Model Config File (Per-Model)

For fine-grained control, create a `model_config` YAML file that specifies different endpoints for different models:

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  fast_model: gpt-4.1-mini
  powerful_model: gpt-4.1

model_settings:
  fast_model:
    api_type: chat_completions
    endpoint: https://openrouter.ai/api/v1
    token: OPENROUTER_API_KEY
  powerful_model:
    api_type: chat_completions
    endpoint: http://localhost:8000/v1
    token: VLLM_API_KEY
```

**Key fields in `model_settings`:**
- `api_type`: `"chat_completions"` (default), `"responses"`, or `"messages"` (Anthropic)
- `endpoint`: Base URL for the API endpoint
- `token`: Name of the environment variable containing the API key (not the key itself)

### Method 3: Command Line Override

Override the model config from the command line:

```bash
python -m seclab_taskflow_agent \
  -t examples.taskflows.echo \
  -m my_custom_model_config
```

## Provider-Specific Examples

### OpenRouter

[OpenRouter](https://openrouter.ai/) provides unified access to multiple LLM providers through a single OpenAI-compatible API.

**1. Get an API key** from [OpenRouter](https://openrouter.ai/keys)

**2. Set the environment variable:**

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

**3. Create a model config:**

```yaml
# examples/model_configs/openrouter.yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  claude: claude-3.5-sonnet
  gpt4: gpt-4-turbo
  llama: meta-llama/llama-3.1-70b-instruct

model_settings:
  claude:
    api_type: chat_completions
    endpoint: https://openrouter.ai/api/v1
    token: OPENROUTER_API_KEY
  gpt4:
    api_type: chat_completions
    endpoint: https://openrouter.ai/api/v1
    token: OPENROUTER_API_KEY
  llama:
    api_type: chat_completions
    endpoint: https://openrouter.ai/api/v1
    token: OPENROUTER_API_KEY
```

**4. Run a taskflow:**

```bash
python -m seclab_taskflow_agent \
  -t examples.taskflows.echo \
  -m examples.model_configs.openrouter
```

**Optional headers:** OpenRouter supports additional headers for analytics and routing. You can set these via environment variables in your taskflow:

```yaml
taskflow:
  - task:
      agents:
        - examples.personalities.assistant
      user_prompt: "Hello"
      env:
        HTTP_REFERER: "https://myapp.example.com"
        X_TITLE: "My Application"
```

### vLLM

[vLLM](https://vllm.readthedocs.io/) is a high-performance inference engine that exposes an OpenAI-compatible API.

**1. Start vLLM server:**

```bash
# Install vLLM
pip install vllm

# Start the server with OpenAI-compatible API
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

**2. Create a model config:**

```yaml
# examples/model_configs/vllm.yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  local_llama: meta-llama/Llama-3.1-8B-Instruct

model_settings:
  local_llama:
    api_type: chat_completions
    endpoint: http://localhost:8000/v1
    token: VLLM_API_KEY  # Can be any non-empty string or empty
```

**3. Set the API key (can be empty for local servers):**

```bash
# vLLM doesn't require authentication by default, but the agent needs a token
export VLLM_API_KEY="dummy"  # or any non-empty string
```

**4. Run a taskflow:**

```bash
python -m seclab_taskflow_agent \
  -t examples.taskflows.echo \
  -m examples.model_configs.vllm
```

**Notes:**
- vLLM supports both `chat_completions` and `responses` API types
- For tool calling, ensure your model supports it (e.g., Llama 3.1 with tool calling fine-tune)
- Adjust `--max-model-len` and `--tensor-parallel-size` based on your model and hardware

### Ollama

[Ollama](https://ollama.ai/) makes it easy to run open-source LLMs locally with an OpenAI-compatible API.

**1. Install and start Ollama:**

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service (runs on http://localhost:11434 by default)
ollama serve
```

**2. Pull a model:**

```bash
ollama pull llama3.1:8b
# or
ollama pull qwen2.5:7b
```

**3. Create a model config:**

```yaml
# examples/model_configs/ollama.yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  llama_local: llama3.1:8b
  qwen_local: qwen2.5:7b

model_settings:
  llama_local:
    api_type: chat_completions
    endpoint: http://localhost:11434/v1
    token: OLLAMA_API_KEY
  qwen_local:
    api_type: chat_completions
    endpoint: http://localhost:11434/v1
    token: OLLAMA_API_KEY
```

**4. Set the API key:**

```bash
# Ollama doesn't require authentication, but the agent needs a token
export OLLAMA_API_KEY="ollama"  # any non-empty string works
```

**5. Run a taskflow:**

```bash
python -m seclab_taskflow_agent \
  -t examples.taskflows.echo \
  -m examples.model_configs.ollama
```

**Notes:**
- Ollama's OpenAI compatibility layer is available at `/v1` endpoint
- For tool calling, use models that explicitly support it (e.g., `llama3.1`, `mistral`, `qwen2.5`)
- Ollama automatically loads models into memory on first request

### Other OpenAI-Compatible Services

Any service that implements the OpenAI Chat Completions API can be used. Here are examples for popular providers:

#### Together AI

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  llama70b: meta-llama/Llama-3.1-70B-Instruct-Turbo

model_settings:
  llama70b:
    api_type: chat_completions
    endpoint: https://api.together.xyz/v1
    token: TOGETHER_API_KEY
```

```bash
export TOGETHER_API_KEY="your-together-api-key"
```

#### Groq

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  llama_fast: llama-3.1-70b-versatile

model_settings:
  llama_fast:
    api_type: chat_completions
    endpoint: https://api.groq.com/openai/v1
    token: GROQ_API_KEY
```

```bash
export GROQ_API_KEY="your-groq-api-key"
```

#### Azure OpenAI

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  gpt4_azure: gpt-4

model_settings:
  gpt4_azure:
    api_type: chat_completions
    endpoint: https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME
    token: AZURE_OPENAI_API_KEY
```

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
```

**Note:** Azure OpenAI uses a different URL structure. You may need to adjust the endpoint to include the deployment name.

#### LM Studio

LM Studio provides a local OpenAI-compatible API server:

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  local_model: local-model

model_settings:
  local_model:
    api_type: chat_completions
    endpoint: http://localhost:1234/v1
    token: LM_STUDIO_API_KEY
```

```bash
export LM_STUDIO_API_KEY="lm-studio"  # any non-empty string
```

## Environment Variable Reference

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `AI_API_ENDPOINT` | Global API endpoint URL | No | `https://api.openai.com/v1` |
| `AI_API_TOKEN` | Global API key | Yes (if no model-specific token) | `sk-...` |
| `COPILOT_TOKEN` | Alternative to `AI_API_TOKEN` | No | `ghp_...` |
| `SECLAB_TASKFLOW_BACKEND` | Backend SDK to use | No | `openai_agents` (default) |
| `MODEL_PARALLEL_TOOL_CALLS` | Enable parallel tool calls | No | `1` to enable |
| `MODEL_TEMP` | Default temperature | No | `0.7` |
| `TASK_AGENT_DEBUG` | Enable debug logging | No | `1` to enable |

**Provider-specific keys:**
- `OPENROUTER_API_KEY` - OpenRouter API key
- `VLLM_API_KEY` - vLLM API key (can be dummy for local)
- `OLLAMA_API_KEY` - Ollama API key (can be dummy for local)
- `TOGETHER_API_KEY` - Together AI API key
- `GROQ_API_KEY` - Groq API key
- `AZURE_OPENAI_API_KEY` - Azure OpenAI API key

## Backend Selection

The Taskflow Agent supports multiple backends. For custom OpenAI-compatible endpoints, use the `openai_agents` backend (default):

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

backend: openai_agents  # Explicit (optional, this is the default)

models:
  my_model: llama-3.1-8b

model_settings:
  my_model:
    endpoint: http://localhost:8000/v1
    token: MY_API_KEY
```

**Available backends:**
- `openai_agents` - OpenAI Agents SDK (default, recommended for custom endpoints)
- `copilot_sdk` - GitHub Copilot SDK (for GitHub Copilot only)
- `anthropic_sdk` - Anthropic SDK (for Anthropic Claude API)

## API Type Configuration

Different endpoints may support different API types:

- `chat_completions` - Standard OpenAI Chat Completions API (`/v1/chat/completions`)
- `responses` - OpenAI Responses API (newer, used by GitHub Copilot)
- `messages` - Anthropic Messages API (`/v1/messages`)

For custom OpenAI-compatible endpoints, use `chat_completions`:

```yaml
model_settings:
  my_model:
    api_type: chat_completions
    endpoint: https://my-provider.com/v1
    token: MY_API_KEY
```

## Multi-Provider Taskflows

You can use multiple providers in a single taskflow by defining different models with different endpoints:

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config

models:
  fast_local: llama3.1:8b
  powerful_cloud: gpt-4-turbo

model_settings:
  fast_local:
    api_type: chat_completions
    endpoint: http://localhost:11434/v1
    token: OLLAMA_API_KEY
  powerful_cloud:
    api_type: chat_completions
    endpoint: https://openrouter.ai/api/v1
    token: OPENROUTER_API_KEY
```

Then in your taskflow, you can use both models:

```yaml
taskflow:
  - task:
      model: fast_local
      agents:
        - examples.personalities.assistant
      user_prompt: "Quick analysis"
  
  - task:
      model: powerful_cloud
      agents:
        - examples.personalities.assistant
      user_prompt: "Deep analysis"
```

## Troubleshooting

### 401 Unauthorized

**Symptoms:** API returns 401 error

**Solutions:**
1. Verify your API key is set correctly:
   ```bash
   echo $AI_API_TOKEN
   echo $OPENROUTER_API_KEY  # or your provider's key
   ```

2. Check that the `token` field in your model config refers to the environment variable name, not the actual key:
   ```yaml
   model_settings:
     my_model:
       token: OPENROUTER_API_KEY  # ✓ Correct - env var name
       # token: sk-or-v1-...      # ✗ Wrong - actual key
   ```

3. Ensure your API key has the required permissions/scopes

### Connection Refused

**Symptoms:** `ConnectionError` or `ConnectionRefusedError`

**Solutions:**
1. Verify the endpoint URL is correct and accessible:
   ```bash
   curl https://your-endpoint.com/v1/models \
     -H "Authorization: Bearer $YOUR_API_KEY"
   ```

2. For local servers (vLLM, Ollama), ensure the service is running:
   ```bash
   # Check if vLLM is running
   curl http://localhost:8000/v1/models
   
   # Check if Ollama is running
   curl http://localhost:11434/v1/models
   ```

3. Check firewall settings and proxy configuration

### Model Not Found

**Symptoms:** API returns 404 or "model not found" error

**Solutions:**
1. List available models from your provider:
   ```bash
   curl https://your-endpoint.com/v1/models \
     -H "Authorization: Bearer $YOUR_API_KEY"
   ```

2. Ensure the model name in your config exactly matches the provider's model ID:
   ```yaml
   models:
     # ✗ Wrong
     my_model: llama-3-8b
     # ✓ Correct (check provider's model list)
     my_model: meta-llama/Llama-3-8B-Instruct
   ```

3. For Ollama, ensure the model is pulled:
   ```bash
   ollama list
   ollama pull llama3.1:8b  # if not listed
   ```

### Tool Calling Not Working

**Symptoms:** Model doesn't call tools or returns errors when tools are provided

**Solutions:**
1. Verify your model supports tool calling. Not all models do. Recommended models:
   - GPT-4, GPT-4-Turbo, GPT-4o
   - Claude 3.5 Sonnet, Claude 3 Opus
   - Llama 3.1 (with tool calling fine-tune)
   - Mistral / Mixtral
   - Qwen 2.5

2. Check that your endpoint supports tool calling. Some providers don't forward tool definitions.

3. Enable debug logging to see what's being sent:
   ```bash
   export TASK_AGENT_DEBUG=1
   python -m seclab_taskflow_agent -t your.taskflow
   ```

4. Try with a known working endpoint first (e.g., OpenAI or OpenRouter) to isolate the issue

### Rate Limiting (429 Errors)

**Symptoms:** API returns 429 Too Many Requests

**Solutions:**
1. Check your provider's rate limits and quota
2. Add delays between requests in your taskflow:
   ```yaml
   taskflow:
     - task:
         # ... task config
     - task:
         # Add a delay before this task
         run: sleep 5
     - task:
         # ... next task
   ```
3. Upgrade your provider plan for higher limits
4. Use multiple API keys and rotate them

### Streaming Not Working

**Symptoms:** Response appears all at once instead of streaming

**Solutions:**
1. Verify your endpoint supports streaming. Most OpenAI-compatible endpoints do, but some don't.
2. Check that you're using `api_type: chat_completions` (streaming is always supported)
3. Some providers may have streaming disabled by default. Check their documentation.

### Incorrect API Type

**Symptoms:** 404 errors or "endpoint not found"

**Solutions:**
1. Ensure you're using the correct `api_type` for your provider:
   - OpenAI-compatible: `chat_completions`
   - GitHub Copilot: `responses` or `chat_completions`
   - Anthropic: `messages` (with `anthropic_sdk` backend)

2. Check the endpoint URL matches the API type:
   ```yaml
   # For chat_completions, endpoint should be base URL
   endpoint: https://api.openai.com/v1
   
   # For responses (GitHub Copilot), use their endpoint
   endpoint: https://api.githubcopilot.com
   ```

## Debugging Tips

### Enable Debug Logging

```bash
export TASK_AGENT_DEBUG=1
python -m seclab_taskflow_agent -t your.taskflow
```

This shows full request/response details including headers and payloads.

### Validate Configuration

Use the `--lint` flag to validate your configuration without making API calls:

```bash
python -m seclab_taskflow_agent --lint -t your.taskflow -m your.model_config
```

### Test Endpoint Connectivity

```bash
# Test basic connectivity
curl -v https://your-endpoint.com/v1/models \
  -H "Authorization: Bearer $YOUR_API_KEY"

# Test chat completions
curl -v https://your-endpoint.com/v1/chat/completions \
  -H "Authorization: Bearer $YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Check Available Models

```bash
curl https://your-endpoint.com/v1/models \
  -H "Authorization: Bearer $YOUR_API_KEY" | jq '.data[].id'
```

## Examples Repository

The `examples/model_configs/` directory contains example configurations for various providers:

- `openrouter.yaml` - OpenRouter multi-provider configuration
- `vllm.yaml` - vLLM local inference server
- `ollama.yaml` - Ollama local LLM runner
- `responses_api.yaml` - GitHub Copilot with Responses API
- `multi_model.yaml` - Multiple models with different settings
- `anthropic_sdk.yaml` - Anthropic Claude configuration
- `copilot_sdk.yaml` - GitHub Copilot SDK configuration

Create your own configurations following these patterns and the examples above.

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [Taskflow Grammar Reference](GRAMMAR.md)
- [Migration Guide](MIGRATION.md)

## Support

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/GitHubSecurityLab/seclab-taskflow-agent/issues)
3. Create a new issue with:
   - Your model config (redact API keys)
   - The full error message (with `TASK_AGENT_DEBUG=1`)
   - Your provider and endpoint URL
   - Steps to reproduce
