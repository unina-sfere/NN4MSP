from importlib import resources

def get_flatland():

    with resources.path("NNforMSP.data", "HVAC_data.csv") as f:
        data_file_path = f
    return data_file_path

