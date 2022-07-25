import numpy as np
import pandas as pd
import chardet
from matplotlib import pyplot as plt
import os
from random import randrange
import random
import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class DataCleaning:
    ''' Class to do initial reading and formatting of the source data file'''

    def __init__(self, filename='DeviceCGM.txt',path='/Users/bsw/Documents/MRPLocal/DATA/'):
        self.path = path
        self.filename = filename
        self.file_source = os.path.join(path, filename)

    def import_and_store(self):
        ''' Input is a txt file with pipe delimiter and specified columns for the specific diabetes dataset
        Returns a list of smaller CSVs (12) shards are created from the original'''
        ## file downloaded from ...
        
        ## detect file encoding
        #with open(path,'rb') as f:
        #  rawdata = b''.join([f.readline() for _ in range(20)])
        #  print(chardet.detect(rawdata)['encoding'])
        ##--> File encoding - UTF-16

        ## Import file to dataframe
        cgmData = pd.read_csv(self.file_source, delimiter="|", encoding='UTF-16')
        
        ## Create shards for easier data management
        fileNames = []
        target = 'CGM_'
        file_write = os.path.join(self.path,target)
        for id, df_i in  enumerate(np.array_split(cgmData, 12)):
            df_i.to_csv(file_write+'{id}.csv'.format(id=id))
            fileNames.append(file_write+'{id}.csv'.format(id=id))

        ## fileNames stores smaller shards of data
        return fileNames

    def openCSV(self,size):
        '''opens up a CSV shard(s) with given filename index and returns a processed dataframe'''
        fileNames=self.import_and_store()
        df = pd.read_csv(fileNames[0])
        for i in range(1,size):
            temp_df = pd.read_csv(fileNames[i])
            df = pd.concat([df,temp_df])
        
        ## Adding features

        df['DeviceDtTm']=pd.to_datetime(df['DeviceDtTm'])
        df['ValueMMOL']=round(df['Value']/18,1)  ## converting to Canadian standard of measurement mmol/L
        df['DDate']=pd.to_datetime(df['DeviceDtTm']).dt.date
        df['hourOfDay'] = df['DeviceDtTm'].dt.hour
        df = df[df['RecordType']=='CGM'] ## remove other record types
        ## ensuring sequence
        df['series']= df['DeviceDtTm'] >= df['DeviceDtTm'].shift() + pd.Timedelta(minutes=6)

        return df

    def resequenceData(self, size=2, filename='/Users/bsw/Documents/MRPLocal/DATA/CGM_Processed.csv'):
        '''resets all the data with the original file from the public dataset
        size is number of shards to include on the new sample'''
        print('Warning - This function will take a while to complete as it is creating sequences based on multiple variables in the data file. Please be patient.')
        dfs = self.openCSV(size)
        dfs.reset_index(inplace=True)
        a = len(dfs)
        seed = np.random.randint(10000,80000)
        curr_ptid = dfs['PtID'].loc[0]
        curr_sequence = seed
        dfs['series_id']=0
        dfs['series_id']=curr_sequence
        for index in range(1,a):
            if dfs['series'].loc[index] == False and dfs['PtID'].loc[index]==curr_ptid:
                dfs['series_id'].loc[index]=curr_sequence
            else:
                curr_sequence+=1
                curr_ptid=dfs['PtID'].loc[index]
                dfs['series_id'].loc[index]=curr_sequence

        dfs.to_csv(filename)



class DataSampling:

    def __init__(self, filename='CGM_Processed.csv',path='/Users/bsw/Documents/MRPLocal/DATA/'):
        self.path = path
        self.filename = filename
        self.file_source = os.path.join(path, filename)
        
        self.samplingDF = pd.read_csv(self.file_source)
        self.samplingDF.drop(['Unnamed: 0.1','index','Unnamed: 0','RecordType','Value'], inplace=True, axis=1)

        ## a few clean-up items
        self.samplingDF['DDate']=pd.to_datetime(self.samplingDF['DDate'])
        self.samplingDF['DeviceDtTm']=pd.to_datetime(self.samplingDF['DeviceDtTm'])
        self.samplingDF.SortOrd=self.samplingDF.SortOrd.astype(int)

        ## cleaning data - removing series where not enough samples to learn / predict
        a=pd.DataFrame((self.samplingDF).groupby(['PtID','series_id'])['RecID'].count())
        a.reset_index(inplace=True)
        self.samplingDF = self.samplingDF[~self.samplingDF.series_id.isin(a.series_id[a.RecID<=25].to_list())] 

    def seriesToTimeSeries(self, X, step_length=8,forecast_dist=6):
        y=[]
        reshapedX = []
        for i in range(len(X)-forecast_dist-step_length):
            y.append(X[i+step_length+forecast_dist])
            reshapedX.append(X[i:i+step_length])
        return reshapedX, y

    def shapeSeriesFromDF(self,df,indexForSelection):
        
        an_X = df[df['series_id']==indexForSelection[0]].ValueMMOL.tolist()
        an_X, y = self.seriesToTimeSeries(an_X)
        X=an_X
        y=y

        for i in indexForSelection[1:]:
            an_X = df[df['series_id']==i].ValueMMOL.tolist()
            an_X, y = self.seriesToTimeSeries(an_X)
            
            X = X+an_X
            y = y+y
        return X,y



    def SampleValidSequences(self, num_clients=8, test_split=0.3,seed=1):
        samplingDF = self.samplingDF
        ## cleaning up the data -- Resetting data types
        
        random.seed(seed)
        
        new_df = samplingDF.groupby('series_id').count()
        ct_df = samplingDF.groupby('PtID').count()

        client_list = ct_df.index.to_numpy()
        cl_ind = client_list[random.sample(range(0,len(client_list)),num_clients)] ##clientids to use for the training

        cl_df = samplingDF[samplingDF.PtID.isin(cl_ind)] ## list of all samples relative to these clients

        series_select = cl_df.groupby('series_id').count()
        series_select = series_select.sample(frac=1)
        series_select = series_select.index.to_list()

        index_cut = int((1-test_split) * len(series_select))
        train_index = series_select[0:index_cut]
        test_index=series_select[index_cut:]

        training_df = samplingDF[samplingDF.series_id.isin(train_index)]
        testing_df = samplingDF[samplingDF.series_id.isin(test_index)]

        ## build training dataset
        X_train,y_train = self.shapeSeriesFromDF(training_df,train_index)
        X_test,y_test = self.shapeSeriesFromDF(testing_df,test_index)

        return X_train, X_test, y_train, y_test

    