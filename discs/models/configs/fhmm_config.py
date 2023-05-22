"""TODO(kgoshvadi): DO NOT SUBMIT without one-line documentation for fhmm_config.

TODO(kgoshvadi): DO NOT SUBMIT without a detailed description of fhmm_config.
"""

from collections.abc import Sequence

from absl import app


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
