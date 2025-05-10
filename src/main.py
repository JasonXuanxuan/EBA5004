import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import regression, aspect_frequency, absa_model

if __name__ == "__main__":
    absa_model()
    # aspect_frequency()
    # regression()