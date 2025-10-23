"""
Basic tests for YAML parsing functionality in the taskflow agent.

Simple parsing + parsing of example taskflows.
"""

import pytest
import tempfile
from pathlib import Path
import yaml
from yaml_parser import YamlParser


class TestYamlParser:
    """Test suite for YamlParser class."""

    def test_yaml_parser_basic_functionality(self):
        """Test basic YAML parsing functionality."""
        # create a temporary directory with test yaml files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_yaml_content = {
                'seclab-taskflow-agent': {
                    'type': 'taskflow',
                    'version': 1
                },
                'taskflow': [
                    {
                        'task': {
                            'agents': ['assistant'],
                            'user_prompt': 'Test prompt'
                        }
                    }
                ]
            }
            
            test_file = temp_path / 'test_taskflow.yaml'
            with open(test_file, 'w') as f:
                yaml.dump(test_yaml_content, f)
            
            parser = YamlParser(temp_path)
            # get all yaml files in the directory
            yaml_files = temp_path.glob('*.yaml')
            result = parser.get_yaml_dict(yaml_files)
            
            assert 'test_taskflow.yaml' in result
            assert result['test_taskflow.yaml']['seclab-taskflow-agent']['type'] == 'taskflow'
            assert len(result['test_taskflow.yaml']['taskflow']) == 1
            assert result['test_taskflow.yaml']['taskflow'][0]['task']['agents'] == ['assistant']


class TestRealTaskflowFiles:
    """Test parsing of actual taskflow files in the project."""

    def test_parse_example_taskflows(self):
        """Test parsing the actual example taskflow files."""
        # this test uses the actual taskflows in the project
        examples_path = Path('taskflows/examples').absolute()
        parser = YamlParser(examples_path)
        
        # Get all YAML files in the examples directory
        yaml_files = examples_path.glob('*.yaml')
        result = parser.get_yaml_dict(yaml_files)
        
        # should contain example files
        assert len(result) > 0

        # check that example.yaml is parsed correctly
        example_task_flow = result['example.yaml']
        assert 'taskflow' in example_task_flow
        assert isinstance(example_task_flow['taskflow'], list)
        assert len(example_task_flow['taskflow']) == 4  # 4 tasks in taskflow
        assert example_task_flow['taskflow'][0]['task']['max_steps'] == 20

    def test_parse_all_taskflows(self):
        """Test parsing all example taskflow files in the project."""
        taskflows_path = Path('taskflows').absolute()
        parser = YamlParser(taskflows_path)
        
        yaml_files = taskflows_path.rglob('*.yaml')
        result = parser.get_yaml_dict(yaml_files)
        
        # should contain all taskflow files (including subdirs)
        assert len(result) == 13
        
        # check access for files with directory structure in names
        example_files = [key for key in result.keys() if 'examples/' in key]
        assert len(example_files) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])