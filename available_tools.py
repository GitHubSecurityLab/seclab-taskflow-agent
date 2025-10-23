import logging

class VersionException(Exception):
    pass

class FileIDException(Exception):
    pass

class FileTypeException(Exception):
    pass

def add_yaml_to_dict(table, key, yaml):
    """Add the yaml to the table, but raise an error if the id isn't unique """
    if key in table:
        raise FileIDException(str(key))
    table.update({key: yaml})

class AvailableTools:
    """
    This class is used for storing dictionaries of all the available
    personalities, taskflows, and prompts.
    """
    def __init__(self, yamls: dict):
        self.personalities = {}
        self.taskflows = {}
        self.prompts = {}
        self.toolboxes = {}
        self.model_config = {}

        # Iterate through all the yaml files and divide them into categories.
        # Each file should contain a header like this:
        #
        #   seclab-taskflow-agent:
        #     type: taskflow
        #     version: 1
        #
        for path, yaml in yamls.items():
            try:
                header = yaml['seclab-taskflow-agent']
                version = header['version']
                if version != 1:
                    raise VersionException(str(version))
                filekey = header['filekey']
                filetype = header['filetype']
                if filetype == 'personality':
                    add_yaml_to_dict(self.personalities, filekey, yaml)
                elif filetype == 'taskflow':
                    add_yaml_to_dict(self.taskflows, filekey, yaml)
                elif filetype == 'prompt':
                    add_yaml_to_dict(self.prompts, filekey, yaml)
                elif filetype == 'toolbox':
                    add_yaml_to_dict(self.toolboxes, filekey, yaml)
                elif filetype == 'model_config':
                    add_yaml_to_dict(self.model_config, filekey, yaml)
                else:
                    raise FileTypeException(str(filetype))
            except KeyError as err:
                logging.error(f'{path} does not contain the key {err.args[0]}')
            except VersionException as err:
                logging.error(f'{path}: seclab-taskflow-agent version {err.args[0]} is not supported')
            except FileIDException as err:
                logging.error(f'{path}: file ID {err.args[0]} is not unique')
            except FileTypeException as err:
                logging.error(f'{path}: seclab-taskflow-agent file type {err.args[0]} is not supported')
