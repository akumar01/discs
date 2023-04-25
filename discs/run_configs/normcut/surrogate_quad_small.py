"""Config for normcut job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='normcut',
          graph_type='nets',
          sweep=[
              {
                  'cfg_str': 'r-VGG',
                  'config.model.stype': ['quad'],
                  'config.experiment.decay_rate': [0.1, 0.05],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [800000],
                  'config.model.penalty': [0.1, 0.01],
                  'config.experiment.init_temperature': [1, 2, 5],
              },
          ],
      )
  )
  return config
