# Python Tests

based on pytest.

## Running Tests

Make sure to install test dependencies: `pip install -r requirements-test.txt`.

### All Tests

```bash
pytest
```


### Specific Test File
```bash
pytest tests/test_yaml_parser.py -v
```

### Specific Test Class
```bash
pytest tests/test_yaml_parser.py::TestYamlParser -v
```

### Specific Test Function
```bash
pytest tests/test_yaml_parser.py::TestYamlParser::test_yaml_parser_basic_functionality -v
```

## Test Configuration

See `pytest.ini` in the root directory.