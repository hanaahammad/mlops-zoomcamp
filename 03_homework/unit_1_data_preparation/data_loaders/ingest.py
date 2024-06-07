import requests
from io import BytesIO
from typing import List

import pandas as pd
import numpy as np

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    '''
    for year, months in [(2024, (1, 3))]:
        for i in range(*months):
            
            response = requests.get(
                'https://github.com/mage-ai/datasets/raw/master/taxi/green'
                f'/{year}/{i:02d}.parquet'
            )
            
            response = requests.get(
                'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
            )
            
            if response.status_code != 200:
                raise Exception(response.text)

            df = pd.read_parquet(BytesIO(response.content))
            df['lpep_pickup_datetime_cleaned'] = df['lpep_pickup_datetime'].astype(np.int64) // 10**9
            

            dfs.append(df)
    '''
    response = requests.get(
        'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')
            
    if response.status_code != 200:
            raise Exception(response.text)

    df = pd.read_parquet(BytesIO(response.content))
    print(df.shape)
    return df
    #return pd.concat(dfs)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'