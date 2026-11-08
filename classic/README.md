# The 2023 version

The first version of this project: a pygame window where a random 40–60% of the board starts alive, and
every live cell is drawn as a skull (`img/skull.png`). It follows Conway's rules (B3/S23) with dead edges,
using a 2-D convolution to count neighbours.

```bash
cd classic
python run_simulation.py      # needs numpy, scipy and pygame
```

Board size and speed are set in `params.py`. The modern simulator in [`lifelab/`](../lifelab) does the
same update with a lookup table on whole stacks of grids, and is tested against a convolution reference
like this one. `lifelab play --skulls` reuses the skull sprite.
