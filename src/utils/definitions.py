import os
from pathlib import Path

def get_project_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    # return Path(__file__).parent.parent 
