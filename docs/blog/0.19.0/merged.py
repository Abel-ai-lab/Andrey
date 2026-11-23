"""Two studies merged into one table: both measure W and Z, only the first measures X, and only the
second measures Y, so no row observes X and Y together."""

import logging
import sys
import textwrap

import numpy as np
import pandas as pd

import andrey


class Wrapped(logging.Formatter):
    """Print each log record as a wrapped paragraph."""

    def format(self, record):
        text = f"{record.levelname}: {record.getMessage()}"
        return textwrap.fill(text, 100, break_on_hyphens=False)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(Wrapped())
logging.getLogger("andrey").addHandler(handler)

# The true graph: X -> Z <- W, and Z -> Y.
rng = np.random.default_rng(0)
n = 2000
x = rng.standard_normal(n)
w = rng.standard_normal(n)
z = 0.8 * x + 0.8 * w + 0.6 * rng.standard_normal(n)
y = 0.8 * z + 0.6 * rng.standard_normal(n)
data = pd.DataFrame({"X": x, "W": w, "Z": z, "Y": y})
data.loc[1000:, "X"] = np.nan  # the second study did not measure X
data.loc[:999, "Y"] = np.nan  # the first study did not measure Y

print("Complete rows:", len(data.dropna()))
print()
out = andrey.pc(data)
print(out)
print()
print(andrey.pc(data, possibly_separable="ambiguous").structure)
print()
try:
    andrey.ges(data, missing="em")
except ValueError as error:
    print("EM:", textwrap.fill(str(error), 96, subsequent_indent="    "))
