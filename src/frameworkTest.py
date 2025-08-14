from dataClasses import Dataset
from analysisClasses import Analysis
import matplotlib.pyplot as plt

# dataset = Dataset(path="/Users/jcapo/cernbox/NLTFdata/ABS/ABS_baseline.xlsx")
# dataset.load()
# dataset.show()
# print(dataset.name)

# dataset.findTimes()
# dataset.purity(show=True)

fig, ax = plt.subplots(figsize=(10, 6))
analysis = Analysis(path="/Users/jcapo/cernbox/NLTFdata/COPPER_TAPE", name="COPPER_TAPE")
# analysis.h2oConcentration(show=True, ax=ax)
# analysis.temperature(show=True, ax=ax)
# analysis.purity(show=True, ax=ax, fit_legend=False)
fig.tight_layout()
plt.show()