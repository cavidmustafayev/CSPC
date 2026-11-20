import os
import pandas as pd 
from ucimlrepo import fetch_ucirepo
concrete=fetch_ucirepo(id=165)
X=concrete.data.features
Y=concrete.data.targets
print(concrete.metadata)
print(concrete.variables)

X=X.copy()
X.columns=["cement", "slag", "ash", "water", "superplastic", "coarse_egg", "fine_agg", "age"]
Y=Y.iloc[:,0].rename("strength")

os.makedirs("PW4/data", exist_ok=True)
X.assign(strength=Y).to_csv("PW4/data/concrete.csv", index=False)
print("Concrete data saved to 'PW4/data/concrete.csv'")