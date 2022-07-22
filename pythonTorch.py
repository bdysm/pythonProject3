import numpy as np
import dask.array as da
x = np.array(range(1000))
x = da.from_array(x, chunks = 10)
da.sin(x).compute()

