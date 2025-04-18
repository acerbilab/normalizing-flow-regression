# for annotating Options as input of itself
from __future__ import annotations

import configparser
import copy
from collections.abc import MutableMapping

import numpy as np


class Options(MutableMapping, dict):
    """
    This class is responsible for Options.

    Parameters
    ----------
    default_options_path : str
        The path to default options that can be overwritten by the user.
    evaluation_parameters : dict
        Parameters used to evaluate the options.
    user_options : dict
        User defined values to overwrite default options.

    Attributes
    ----------
    useroptions : set
        This set contains all options that have set by the user,
        if there are none it is empty. These ``useroptions`` are immutable to
        changes using :py:meth:`load_options_file`.
    """

    def __init__(
        self,
        default_options_path: str,
        evaluation_parameters: dict = None,
        user_options: dict = None,
    ):
        """
        Initialize the options using default options and specified options from
        the user.
        """
        # Flag initialization as in-progress
        # (completed in self.validate_option_names)
        self.is_initialized = False
        super().__init__()
        self.descriptions = dict()
        self["useroptions"] = set()

        self.default_options_path = default_options_path
        self.evaluation_parameters = evaluation_parameters
        self.user_options = user_options

        # load options from file
        self.load_options_file(default_options_path, evaluation_parameters)

        # User options
        if user_options is not None:
            self.update(user_options)
            self["useroptions"].update(user_options.keys())

    @classmethod
    def init_from_existing_options(
        cls,
        default_options_path: str,
        evaluation_parameters: dict = None,
        other: Options = None,
    ):
        """
        Initialize an options instance using default options and another options
        instance.

        Only the user-definied options from the other object will overwrite the
        default options. Everything else will come from the default options.

        Parameters
        ----------
        default_options_path : str
            The path to default options that can be overwritten by the user.
        evaluation_parameters : dict
            Parameters used to evaluate the options.
        other : Options
            User defined values to overwrite default options.

        Returns
        -------
        new_options : Options
            The new options object with the values merged as described above.
        """
        if other is None:
            user_options = None
        else:
            user_option_keys = other.get("useroptions")
            user_options = {k: other.get(k) for k in user_option_keys}
        new_options = cls(
            default_options_path, evaluation_parameters, user_options
        )
        return new_options

    def load_options_file(
        self, options_path: str, evaluation_parameters: dict = None
    ):
        """
        Load options from an ini file and evaluate them using the specified
        ``evaluation_parameters``.

        Note that strings starting with # in the .ini file act as description to
        the option in the following line.

        Parameters
        ----------
        options_path : str
            The path to an options ini file that should be loaded.
        evaluation_parameters : dict, optional
            Parameters used to evaluate the options.
        """

        # evaluation_parameters
        for key, val in evaluation_parameters.items():
            exec(key + "=val")

        options_list = _read_config_file(options_path)
        for key, value, description in options_list:
            if key not in self.get("useroptions") and key != "useroptions":
                self[key] = eval(value)
                self.descriptions[key] = description

    def validate_option_names(self, options_paths: list):
        """
        Check that ini files specified by the list of ``options_paths`` contain
        the option names from this options object at least once.

        Note that this method checks not if there are option names in files that
        are not in the object. After option names are validated, initialization
        is flagged as complete and `self.is_initialized` is set to `True` to
        prevent further modification of options.

        Parameters
        ----------
        options_paths : list of str
            A list of paths to ini files can contain the allowed option names.

        Raises
        ------
        ValueError
            Raised when an option exists in this object but not in one of the
            specified ini files.
        """
        # create set of option names from all ini files
        file_option_names = set()
        for options_path in options_paths:
            file_option_names.update(
                _read_config_file(options_path)[:, 0].flatten()
            )

        for key in self.keys():
            if key != "useroptions" and key not in file_option_names:
                raise ValueError(f"The option {key} does not exist.")

        # After initialzation is complete prevent changes to options:
        self.is_initialized = True

    def __setitem__(self, key, val, force=False):
        # Prevent user from attempting to modify options after initialization
        if self.is_initialized and not force:
            raise AttributeError(
                "Warning: Cannot set options after initialization. Please re-initialize with `user_options = {...}`"
            )
        else:
            dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.default_options_path,
                self.evaluation_parameters,
                self.user_options,
            ),
            self.__getstate__(),
        )

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        # Copy class properties:
        result.__dict__.update(self.__dict__)
        # Copy options dict:
        for k, v in dict.items(self):
            result.__setitem__(k, v, force=True)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        # Avoid infinite recursion in deepcopy
        memo[id(self)] = result
        # Copy class properties:
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        # Copy options dict:
        for k, v in dict.items(self):
            result.__setitem__(k, copy.deepcopy(v, memo), force=True)
        return result

    def eval(self, key: str, evaluation_parameters: dict):
        """
        Evaluate an option using `evaluation_parameters` if it is a callable,
        otherwise return the value of the option.

        Parameters
        ----------
        key : str
            The name of the option.
        evaluation_parameters : dict
            Parameters for the options in case it is a callable. These have to
            match the key arguments of the callable and are ignored if it is not
            a callable.

        Returns
        -------
        val : object
            Value of the object which has been evaluated if it is a callable.
        """
        if callable(self.get(key)):
            return self.get(key)(**evaluation_parameters)
        else:
            return self.get(key)

    def __str__(self):
        """
        Returns the options in a format key: value (description).

        Returns
        -------
        str
            The str to describe an options object.
        """
        return "".join(
            [
                f"{k}: {v} ({str(self.descriptions.get(k))}) \n"
                for (k, v) in self.items()
            ]
        )


def _read_config_file(options_path: str):
    """
    Private helper method to read a config file and return the options as a
    list of tuples (key, value, description).

    Note that strings starting with # in the .ini file act as description to
    the option in the following line.
    """
    conf = configparser.ConfigParser(comment_prefixes="", allow_no_value=True)
    # do not lower() both values as well as descriptions
    conf.optionxform = str
    conf.read(options_path)

    option_list = list()
    description = ""
    for section in conf.sections():
        for key, value in conf.items(section):
            if "#" in key:
                description = key.strip("# ")
            else:
                option_list.append([key, value, description])
                description = ""

    if len(option_list) == 0:
        raise ValueError(
            f"The option file at {options_path} does not contain options."
        )

    return np.array(option_list)
