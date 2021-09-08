import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv("db/db-gal-negative.csv")
df['Label'] = 0
df.to_csv("db/db-negative-label.csv", index=False)

df2= pd.read_csv("db/db-positive.csv")
df2['Label'] = 1
df2.to_csv("db/db-positive-label.csv", index=False)

frames = [df, df2]
result = pd.concat(frames, ignore_index=True)
result = result.to_numpy()
np.random.shuffle(result)
result_df =pd.DataFrame(data=result, columns=["Post", "Label"])
result_df.to_csv("DomecticViolence.csv")
