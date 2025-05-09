import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import regression, aspect_frequency

if __name__ == "__main__":
    aspect_frequency()
    # regression()