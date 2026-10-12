# %%
import localbenchmark
print(dir(localbenchmark))
print(localbenchmark.__version__)

# %%
from localbenchmark import localbenchmark
model = localbenchmark()
model.plot()

# %%
