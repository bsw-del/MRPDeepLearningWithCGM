# MRPDeepLearningWithCGM
Deep Learning applications to Blood Glucose forecasting

## Dataprep.py

File that will prepare the data for the project (for use with DL algorithms). 
<br>
`DataCleaning` has required methods to setup and structure the data from the raw inputs downloaded from https://public.jaeb.org/datasets/diabetes -- CityPublicDataset.zip (file: DeviceCGM.txt) <br>
`resequenceData` will create a single CSV with the data cleaned and sequenced based on if the timeseries was complete and without gaps.