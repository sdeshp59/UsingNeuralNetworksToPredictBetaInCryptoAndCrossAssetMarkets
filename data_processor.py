from dataclasses import dataclass, field
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import random
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Dict, List, Tuple, Optional
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')


class DataProcessor():
    def __init__(self, with_ff:bool=False):
        self.with_ff = with_ff    
        
    def run(self) -> pd.DataFrame:
        df = self._load_asset_data()
        df = self._clean_asset_data(df)
        df["split"] = df.index.to_period("M").map(self._month_split) # type: ignore
        if self.with_ff:
            ff = self._load_ff()
            ff = self._clean_ff(ff)
            df = self._merge_ff(df, ff)
        return df
    

    def _load_asset_data(self)->pd.DataFrame:
        bitcoin = pd.read_csv('data/CBBTCUSD.csv')
        bitcoin['observation_date'] = bitcoin['observation_date'].astype(str)

        ethereum = pd.read_csv('data/CBETHUSD.csv')
        ethereum['observation_date'] = ethereum['observation_date'].astype(str)

        litecoin= pd.read_csv('data/CBLTCUSD.csv')
        litecoin['observation_date'] = litecoin['observation_date'].astype(str)

        bitcoin_cash= pd.read_csv('data/CBBCHUSD.csv')
        bitcoin_cash['observation_date'] = bitcoin_cash['observation_date'].astype(str)

        fiat= pd.read_csv('data/DTWEXBGS.csv')
        fiat['observation_date'] = fiat['observation_date'].astype(str)

        gold= pd.read_csv('data/NASDAQQGLDI.csv')
        gold['observation_date'] = gold['observation_date'].astype(str)

        vde= pd.read_csv('data/VDE.csv')
        vde['Date'] = vde['Date'].astype(str)
        
        msf = pd.read_csv('data/MSF_1996_2023.csv')
        msf['date']=msf['date'].astype(str)
        
        btc_clean = self._prepare_series(bitcoin, 'observation_date', 'CBBTCUSD', treat_as_price=True)
        eth_clean = self._prepare_series(ethereum, 'observation_date', 'CBETHUSD', treat_as_price=True)
        ltc_clean = self._prepare_series(litecoin, 'observation_date', 'CBLTCUSD', treat_as_price=True)
        bch_clean = self._prepare_series(bitcoin_cash, 'observation_date', 'CBBCHUSD', treat_as_price=True)
        msf_clean = self._prepare_series(msf, 'date', 'vwretd', treat_as_price=False)
        fiat_clean = self._prepare_series(fiat, 'observation_date', 'DTWEXBGS', treat_as_price=True)
        gold_clean = self._prepare_series(gold, 'observation_date', 'NASDAQQGLDI', treat_as_price=True)
        vde_clean = self._prepare_series(vde, 'Date', 'Close', treat_as_price=True)
        
        all_series = {
            'BTC': btc_clean,
            'ETH': eth_clean,
            'LTC': ltc_clean,
            'BCH': bch_clean,
            'FIAT': fiat_clean,
            'GOLD': gold_clean,
            'VDE': vde_clean,
            'MSF': msf_clean
        }

        for name, df_series in all_series.items():
            df_series.rename(columns={
                df_series.columns[0]: f'{name}_price',
                'pct_ret': f'{name}_pct_ret'
            }, inplace=True)

        df = pd.concat(all_series.values(), axis=1, join='outer')
        df.sort_index(inplace=True)
        df = df.loc['2015-01':'2023-12']
        return df
    
    def _prepare_series(self, df, date_col, value_col, how='last', treat_as_price=True):
        df = df[[date_col, value_col]]
        df[date_col] = pd.to_datetime(df[date_col])
        s = df.groupby(date_col)[value_col].agg(how).sort_index()
        daily_idx = pd.date_range(s.index.min(), s.index.max(), freq='D')
        s = s.reindex(daily_idx).ffill()
        monthly = s.resample('M').last().to_frame(name=value_col)
        
        if treat_as_price:
            monthly['pct_ret'] = monthly[value_col].pct_change()
        else:
            monthly['pct_ret'] = monthly[value_col]
        
        return monthly
    
    def _clean_asset_data(self, df):
        market_cols_to_fill = [
            'FIAT_price', 'FIAT_pct_ret', 
            'GOLD_price', 'GOLD_pct_ret', 
            'VDE_price', 'VDE_pct_ret', 
            'MSF_price', 'MSF_pct_ret'
        ]
        df[market_cols_to_fill] = df[market_cols_to_fill].ffill()

        crypto_ret_cols = ['BTC_pct_ret', 'ETH_pct_ret', 'LTC_pct_ret', 'BCH_pct_ret']
        crypto_available = df[crypto_ret_cols].notna().sum(axis=1)
        df = df[crypto_available >= 3]

        df[crypto_ret_cols] = df[crypto_ret_cols].fillna(0.0)

        df = df.dropna(subset=['MSF_pct_ret', 'FIAT_pct_ret', 'GOLD_pct_ret', 'VDE_pct_ret'])
        return df
    
    def _load_ff(self)-> pd.DataFrame:
        df = pd.read_csv('data/FF_5Factor.csv')
        return df
    
    def _clean_ff(self, df:pd.DataFrame)->pd.DataFrame:
        df["Unnamed: 0"] = df["Unnamed: 0"].astype(str).str.strip()
        valid_ym = df["Unnamed: 0"].str.fullmatch(r"\d{6}")
        df = df.loc[valid_ym].copy()  
        df["date"] = pd.to_datetime(df["Unnamed: 0"], format="%Y%m").dt.to_period("M")
        df = df.drop(columns=["Unnamed: 0"])
        for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        df["RF"] = df["RF"] / 100.0
        df = df[["date","Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]]
        return df
    
    def _merge_ff(self, df: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = df.index.to_period("M") # type: ignore
        ff = ff.set_index("date")
        out = df.join(ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]], how="left")
        out = out.rename(columns={"RF": "rf"})
        for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "rf"]:
            out[col] = out[col].ffill()
        return out
    
    def make_lagged_features_for_model(self, df: pd.DataFrame, crypto_ret_col: str,
                                        market_ret_col: str, w: int, with_ff: bool = False,
                                        ff_factors: List[str] = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "rf"]): # type: ignore
        main = df.sort_index().copy()
        feat_cols = []

        if with_ff and ff_factors is None:
            ff_factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "rf"]

        for k in range(1, w + 1):
            main[f'crypto_lag{k}'] = main[crypto_ret_col].shift(k)
            feat_cols.append(f'crypto_lag{k}')
            main[f'market_lag{k}'] = main[market_ret_col].shift(k)
            feat_cols.append(f'market_lag{k}')

        if with_ff:
            for factor in ff_factors:
                for k in range(1, w + 1):
                    col_name = f'{factor}_lag{k}'
                    main[col_name] = main[factor].shift(k)
                    feat_cols.append(col_name)

        main['crypto_ret_next'] = main[crypto_ret_col].shift(-1)
        main['market_ret_next'] = main[market_ret_col].shift(-1)
        main = main.dropna(subset=['crypto_ret_next', 'market_ret_next'])
        main[feat_cols] = main[feat_cols].fillna(0.0)

        return main, feat_cols

    @staticmethod
    def _month_split(ym:pd.Period):
        if pd.Period("2015-01", "M") <= ym <= pd.Period("2019-12", "M"):
            return "train"
        elif pd.Period("2020-01", "M") <= ym <= pd.Period("2021-12", "M"):
            return "val"
        else:
            return "test"