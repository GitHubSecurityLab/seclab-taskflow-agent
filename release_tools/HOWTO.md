# How to release the Agent and its Docker image

To release an updated version of the Agent perform the following steps:

1. Add any newly created files or dependencies to `release.txt`.

2. Release an updated Docker image:

```sh
docker login ghcr.io -u YOUR_GITHUB_USERNAME
python release_tools/publish_docker.py release.txt main.py ghcr.io/githubsecuritylab/seclab-taskflow-agent latest
```

Note: your login password is a GitHub PAT with packages write/read/delete scope enabled.

# Notes on our Docker image configuration

For simplicity we use a single Dockerfile that contains all the dependencies required for both our Agent and our various MCP servers.

Since we provide a mount path for the main agent that is configurable via an environment variable, you can provide custom data to the included stdio MCP servers without any Docker image requirements. By setting a path in the `MY_DATA` environment variable, that data will be available in `/app/my_data` to the Agent and its included MCP servers.

Likewise you can mount custom taskflows (`MY_TASKFLOWS`), personalities (`MY_PERSONALITIES`), and prompts (`MY_PROMPTS`) into the Docker image to make them available for use by the Agent.

See `docker/run.sh` for details on how to leverage those configurations. We do also provide the host Docker socket to the image such that 3rd party Docker MCP server images, such as the GitHub MCP server, work as expected.

The default entry point for our Agent Docker image is `/app/main.py`. If you'd like to deploy one of our MCP servers as a standalone server via the Docker image, use `--entrypoint` to set the appropriate entry point.

For example, a configuration to run the echo MCP server via Docker image instead, would look like:

```yaml
server_params:
  kind: stdio
  command: docker
  args: ["run", "--entrypoint", "python" "-i", "--rm", "ghcr.io/githubsecuritylab/seclab-taskflow-agent", "toolboxes/mcp_servers/echo/echo.py"]
```

# How to test and release new PyPI package

See [the packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/#namespace-packages).

We need all the code to be in a separate directory to build it into a package, so we create a new dir and copy what is need for the build.

For pacakges, only underscores are a allowed as legal python idnetifiers and not dashes, so we need to rename the folder.
```bash
mkdir taskflow-package
cd taskflow-package
git clone https://github.com/GitHubSecurityLab/seclab-taskflow-agent.git
mv seclab-taskflow-agent seclab_taskflow_agent
```

Build instructions are in pyproject.toml, and we also need .gitignore to ignore the `venv`.
```bash
cp seclab_taskflow_agent/pyproject.toml pyproject.toml
cp seclab_taskflow_agent/.gitignore .gitignore

# Create a new python virtual env, activate it, and install the tools for the build.
python -m venv venv
source venv/bin/activate

pip install --upgrade build
pip install hatch-requirements-txt

python -m build
pip install --upgrade twine

```

This will create a `dist` directory with a tar.gz source distribution and whl built distribution.

You can test if the package works without uploading it to PyPI by installing it with the whl. Use ` --force-reinstall` if you made a new version of the package. We use `pydantic_core` in the deps, which doesn't seem to work on some versions of macos due to Rust bindings.
```bash
pip install dist/seclab_taskflow_agent-0.0.1-py3-none-any.whl
```

Create an .env file with `COPILOT_TOKEN`, and run the package with:
```bash
python -m seclab_taskflow_agent -p assistant 'how do modems work'
```

To upload it to TestPyPI (you'll need [an account on testpypi and an API token](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives)). Note if you then try to download the package from TestPyPI and run it, it won't work, because TestPyPi does not have the dependencies that are required for seclab-taskflow-agent. New packages on TestPyPI are regularly cleared. Test it instead using the wheel, or by using PyPI.
```bash
python -m twine upload --repository testpypi dist/*
```

To upload it on PyPI (you'll need [an account on PyPI and an API token](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives)). Note you need to update pyproject.toml to a new (higher) version.
```bash
python -m twine upload dist/*
```

Create a fresh venv, and download the package:
```bash
python -m venv .venv
source .venv/bin/activate
pip install seclab-taskflow-agent
```

Create an .env file with `COPILOT_TOKEN`, and run the package with:
```bash
python -m seclab_taskflow_agent -p assistant 'how do modems work'
```
