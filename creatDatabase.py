import numpy as np
import pandas as pd
"""
This scripts creates a combined DomesticViolence database from
the negative posts database (not critical posts)
the positive posts database (critical posts)
The combined database combine all the posts in a random way and is saved to
"DomesticViolenceDataBase.csv". each time you will run this script you will
get a database that is organized differently beacause of the randomness.
"""
dfNegative = pd.read_csv("db/db-negative-label.csv")
dfNegative['Label'] = 0

dfPositive= pd.read_csv("db/db-positive-label.csv")
dfPositive['Label'] = 1


frames = [dfNegative, dfPositive]
result = pd.concat(frames, ignore_index=True)
result = result.to_numpy()
np.random.shuffle(result)
result_df =pd.DataFrame(data=result, columns=["Post", "Label"])
result_df.to_csv("DomesticViolenceDataBase.csv")
