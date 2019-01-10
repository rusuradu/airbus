import pandas as pd


pd.read_csv('../ShipDetection/sample_submission_v2.csv', converters={'EncodedPixels':lambda p:None}).to_csv('basemodel.csv', index=False)
