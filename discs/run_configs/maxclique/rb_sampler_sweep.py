from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='maxclique',
          sampler='path_auxiliary',
          graph_type='rb',
          sweep=[
              {
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'config.experiment.decay_rate': [0.1],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.batch_size': [16],
                  'config.experiment.init_temperature': [1.0],
              },
              {
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                      'dlmc',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'config.experiment.decay_rate': [0.1],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.batch_size': [16],
                  'config.experiment.init_temperature': [1.0],
              },
          ],
      )
  )
  return config
