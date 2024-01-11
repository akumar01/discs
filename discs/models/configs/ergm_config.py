"""Config file for ERGM."""
from ml_collections import config_dict



def get_config():
    model_config = dict(
      shape=(10,10),
      num_categories=2,
      name='ergm',
    )


    return config_dict.ConfigDict(model_config)
