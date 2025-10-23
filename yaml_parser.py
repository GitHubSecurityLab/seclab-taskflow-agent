import os
import yaml
from pathlib import Path
from collections.abc import Iterator
import logging

class YamlParser:
    def __init__(self, rootpath: Path):
        self.rootpath = rootpath.absolute()

    def get_yaml_dict(self, files: Iterator) -> dict:
        """Return a directory of yaml files as a dictionary of file name indexed yaml dicts"""
        yaml_dict = {}
        for f in files:
            if f.is_file() and not f.name.startswith('.') and f.suffix in ['.yaml', '.yml']:
                try:
                    with open(f) as stream:
                        try:
                            p = yaml.safe_load(stream)
                            # Use the relative path as the key in the dict.
                            namespaced_stem = str(f.relative_to(self.rootpath))
                            yaml_dict[namespaced_stem] = p
                        except yaml.YAMLError as e:
                            logging.error(e)
                except FileNotFoundError as e:
                    # deal with editor temp files etc. that might have disappeared
                    logging.error(f"File not found (editor tmp file?): {e}")
                    continue
        return yaml_dict
