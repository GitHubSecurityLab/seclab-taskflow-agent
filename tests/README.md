# Python Tests

Tests are managed via [Hatch](https://hatch.pypa.io/) and run with pytest.

## Running Tests

### All Tests

```bash
hatch test
```

### Target a specific Python version

```bash
hatch test --python 3.11
hatch test --python 3.13
```

### Specific Test File

```bash
hatch test -- tests/test_yaml_parser.py -v
```

### Specific Test Class

```bash
hatch test -- tests/test_yaml_parser.py::TestYamlParser -v
```

### Specific Test Function

```bash
hatch test -- tests/test_yaml_parser.py::TestYamlParser::test_yaml_parser_basic_functionality -v
```

### Full CI invocation

```bash
hatch fmt --linter --check
hatch test --python 3.11 --cover --randomize --parallel --retries 2 --retry-delay 1
```

## Test Configuration

See `[tool.hatch]` and `[tool.coverage]` sections in `pyproject.toml`.