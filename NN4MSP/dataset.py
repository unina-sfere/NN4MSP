import pkg_resources
import pandas as pd

def load_HVAC_data():
    """Return HVAC data set"""

    stream = pkg_resources.resource_stream(__name__, 'data/HVAC_data.csv')
    return pd.read_csv(stream, encoding='latin-1')

