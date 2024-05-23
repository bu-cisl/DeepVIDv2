"""
https://github.com/AllenInstitute/deepinterpolation/blob/master/deepinterpolation/generic.py
"""

import json


class JsonLoader:
    """
    JsonLoader is used to load the data from all structured json files associated with the DeepInterpolation package.
    """

    def __init__(self, path):
        self.path = path

        self.load_json()

    def load_json(self):
        """
        This function load the json file from the path recorded in the class instance.
        Parameters:
        None
        Returns:
        None
        """

        with open(self.path, "r") as f:
            self.json_data = json.load(f)

    def set_default(self, parameter_name, default_value):
        """
        set default forces the initialization of a parameter if it was not present in
        the json file. If the parameter is already present in the json file, nothing
        will be changed.
        Parameters:
        parameter_name (str): name of the paramter to initialize
        default_value (Any): default parameter value
        Returns:
        None
        """

        if not (parameter_name in self.json_data):
            self.json_data[parameter_name] = default_value

    def get_type(self):
        """
        json types define the general category of the object the json file applies to.
        For instance, the json can apply to a data Generator type
        Parameters:
        None

        Returns:
        str: Description of the json type
        """

        return self.json_data["type"]

    def get_name(self):
        """
        Each json type is sub-divided into different names. The name defines the exact construction logic of the object and how the
        parameters json data is used. For instance, a json file can apply to a Generator type using the AudioGenerator name when
        generating data from an audio source. Type and Name fully defines the object logic.
        Parameters:
        None

        Returns:
        str: Description of the json name
        """

        return self.json_data["name"]


class JsonSaver:
    """
    JsonSaver is used to save dict data into individual file.
    """

    def __init__(self, dict_save):
        self.dict = dict_save

    def save_json(self, path):
        """
        This function save the json file into the path provided.
        Parameters:
        str: path: str
        Returns:
        None
        """
        if isinstance(self.dict, dict):
            with open(path, "w") as f:
                json.dump(self.dict, f, indent=4)
        elif isinstance(self.dict, list):
            with open(path, "w") as f:
                for line in self.dict:
                    json.dump(line, f, indent=4)
                    f.write("\n")
