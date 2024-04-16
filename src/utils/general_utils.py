"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
import random
import torch
from typing import List, Dict


def fix_random_seed(random_seed: int) -> None:
    """fix random seeds

    Args:
        seed (int): random seed
    
    Returns:
        None
    """
    # Set a random seed for Python's random module
    random.seed(random_seed)

    # Set a random seed for NumPy
    np.random.seed(random_seed)

    # Set a random seed for PyTorch on both CPU and GPU (if available)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Additional steps for ensuring reproducibility in data loading if using DataLoader
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def update_module_step(step: int, modules: List) -> None:
    """ update module step number. Require modules element to have 'self.update_step(step)' function

    Args:
        step (int): step number
        modules (List): module to update step.
    """
    for module in modules:
        module.update_step(step)


class InfoPrinter():
    def __init__(self, 
                 method    : str = None,
                 total_step: int = 0,
                 scene     : str = None,
                 ):
        """ initialize information printer
    
        Args:
            method (str)    : method name
            total_step (int): total number of iterations
            scene (str)     : scene name
    
        Attributes:
            method (str)    : method name
            str_len (int)   : limit the string within this length
            total_step (int): total number of iterations
            scene (str)     : scene name
            
        """
        self.method = method
        self.str_len = 20
        self.total_step = total_step
        self.scene = scene
    
    def update_total_step(self, total_step: int) -> None:
        """ update total step
    
        Args:
            total_step (int): total step
    
        Attributes:
            total_step (int): total step
        """
        self.total_step = total_step
    
    def update_scene(self, scene: str) -> None:
        """ update scene
    
        Args:
            scene (str): total step
    
        Attributes:
            scene (str): total step
        """
        self.scene = scene
    
    def adjust_string_length(self, desired_length: int, input_str: str) -> str:
        """
        Adjusts the length of a string to a specified length by padding with spaces or cutting the string.

        Args:
            desired_length (int): The desired length of the string.
            input_str (str)     : The input string to adjust.

        Returns:
            str: The adjusted string with the specified length.
        """
        # If the input string is shorter than the desired length, pad it with spaces
        if len(input_str) < desired_length:
            return input_str.ljust(desired_length)
        # If the input string is longer than the desired length, cut it to fit
        else:
            return input_str[:desired_length]
    
    def print(self,
            msg   : str = "",
            step  : int = None,
            module: str = None,
            ) -> str:
        """ print information in the format of 
        | [{self.method}] | Step-{step} | {module}   | {msg}
    
        Args:
            msg (str)   : message string
            step (int)  : step
            module (str): module name
    
        """
        info = f"| [{self.method}] | "
        if self.scene is not None:
            info += f"{self.scene} | "
        if step is not None:
            info += f"Step: {step:04} / {self.total_step:04} | "
        if module is not None:
            module_str = self.adjust_string_length(self.str_len, module)
            info += f"{module_str} | "
        info += msg
        print(info)
    
    def __call__(self, *args, **kwargs):
        self.print(*args, **kwargs)


def update_results_file(results: Dict[str, float], file_path: str) -> None:
    """Update or append key-value pairs to a results text file.

    Args:
        results (Dict[str, float]): A dictionary with string keys and float values.
        file_path (str)           : The path to the text file storing the results.

    """
    ### Read existing results from the file ###
    existing_results = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(',') 
                existing_results[key] = float(value)
    except FileNotFoundError:
        ### If the file does not exist, we'll create it later ###
        pass

    ### Update existing results with new ones or append new ones if they don't exist ###
    existing_results.update(results)

    ### Write the updated results back to the file ###
    with open(file_path, 'w') as file:
        for key, value in existing_results.items():
            file.write(f'{key},{value}\n')
