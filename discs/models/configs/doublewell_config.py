"""Config file for ising model."""
from ml_collections import config_dict


def get_config():
  model_config = dict(
      dim=10,
      range=2,
      temp=1,
      offset=40,
      name='double_well'
  )

  model_config['save_dir_name'] = 'double_well'

  return config_dict.ConfigDict(model_config)
