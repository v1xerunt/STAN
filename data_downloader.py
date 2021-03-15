import logging
import os
from datetime import datetime
from multiprocessing import Pool

import pandas as pd

from utils import get_data_location, download_data


class GenerateTrainingData:

    def __init__(self):
        self.df = None
        self.url_base = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/" \
                        f"csse_covid_19_daily_reports_us/"
        self.common_columns = ["state", "latitude", "longitude", "fips", "date_today", "confirmed", "deaths",
                               "recovered",
                               "active", "hospitalization"]

    def download_single_file(self, date):
        url = self.url_base + "/" + f"{date}.csv"
        #url = os.path.join(self.url_base, f"{date}.csv")
        data = download_data(url=url)
        if data is None:
            logging.info(f"{date}.csv doesn't not exists or failed to be downloaded!")
            return None
        data.loc[:, 'date_today'] = datetime.strptime(date, "%m-%d-%Y")
        data = data.rename(columns={"Province_State": "state", "Lat": "latitude", "Long_": "longitude",
                                    'Confirmed': "confirmed", 'Deaths': "deaths", 'Recovered': "recovered",
                                    'Active': "active", 'FIPS': "fips", "People_Hospitalized": "hospitalization"}) \
            .dropna(subset=['fips'])
        data.loc[:, "fips"] = data['fips'].astype(int)
        data = data[self.common_columns].fillna(0)
        return data

    def download_jhu_data(self, start_time, end_time):
        date_list = pd.date_range(start_time, end_time).strftime("%m-%d-%Y")
        data = Pool().map(self.download_single_file, date_list)
        print('Finish download')
        data = [x for x in data if x is not None]
        data = pd.concat(data, axis=0)
        data.loc[:, 'date_today'] = pd.to_datetime(data['date_today'])
        df = []
        for fips in data['fips'].unique():
            temp = data[data['fips'] == fips].sort_values('date_today')
            temp.loc[:, "new_cases"] = temp['confirmed'].copy()
            # transform to daily cases
            for col in ["new_cases", "deaths", "hospitalization"]:
                t = temp[col].copy().sort_values().to_numpy()
                t[1:] = t[1:] - t[:-1]
                temp = temp.iloc[1:]
                temp.loc[:, col] = t[1:]
            df.append(temp)
        df = pd.concat(df, axis=0)
        df.to_pickle(get_data_location('state_covid_data.pickle'))
        return df
