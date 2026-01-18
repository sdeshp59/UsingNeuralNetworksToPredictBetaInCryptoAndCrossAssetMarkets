import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data_processor import DataProcessor


class TestDataProcessorInit:
    def test_init_default(self):
        dp = DataProcessor()
        assert dp.with_ff == False

    def test_init_with_ff_true(self):
        dp = DataProcessor(with_ff=True)
        assert dp.with_ff == True

    def test_init_with_ff_false(self):
        dp = DataProcessor(with_ff=False)
        assert dp.with_ff == False


class TestMonthSplit:
    def test_train_period_start(self):
        period = pd.Period("2015-01", "M")
        assert DataProcessor._month_split(period) == "train"

    def test_train_period_end(self):
        period = pd.Period("2019-12", "M")
        assert DataProcessor._month_split(period) == "train"

    def test_train_period_middle(self):
        period = pd.Period("2017-06", "M")
        assert DataProcessor._month_split(period) == "train"

    def test_val_period_start(self):
        period = pd.Period("2020-01", "M")
        assert DataProcessor._month_split(period) == "val"

    def test_val_period_end(self):
        period = pd.Period("2021-12", "M")
        assert DataProcessor._month_split(period) == "val"

    def test_val_period_middle(self):
        period = pd.Period("2020-06", "M")
        assert DataProcessor._month_split(period) == "val"

    def test_test_period(self):
        period = pd.Period("2022-01", "M")
        assert DataProcessor._month_split(period) == "test"

    def test_test_period_later(self):
        period = pd.Period("2023-06", "M")
        assert DataProcessor._month_split(period) == "test"


class TestPrepareSeries:
    def test_prepare_series_price_data(self):
        dp = DataProcessor()
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-15', '2020-02-01', '2020-02-15'],
            'price': [100.0, 110.0, 120.0, 130.0]
        })
        result = dp._prepare_series(df, 'date', 'price', treat_as_price=True)

        assert 'price' in result.columns
        assert 'pct_ret' in result.columns
        assert len(result) == 2  # Two months
        assert result['pct_ret'].iloc[1] == pytest.approx((130.0 - 110.0) / 110.0, rel=1e-5)

    def test_prepare_series_return_data(self):
        dp = DataProcessor()
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-15', '2020-02-01', '2020-02-15'],
            'ret': [0.01, 0.02, 0.03, 0.04]
        })
        result = dp._prepare_series(df, 'date', 'ret', treat_as_price=False)

        assert 'ret' in result.columns
        assert 'pct_ret' in result.columns
        # When treat_as_price=False, pct_ret should equal the value column
        assert result['pct_ret'].iloc[0] == result['ret'].iloc[0]

    def test_prepare_series_forward_fill(self):
        dp = DataProcessor()
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-10'],
            'price': [100.0, 150.0]
        })
        result = dp._prepare_series(df, 'date', 'price', treat_as_price=True)

        # Should have one month (January 2020)
        assert len(result) == 1
        # Last value of January should be 150.0
        assert result['price'].iloc[0] == 150.0


class TestCleanFF:
    def test_clean_ff_valid_data(self):
        dp = DataProcessor()
        df = pd.DataFrame({
            'Unnamed: 0': ['202001', '202002', '202003'],
            'Mkt-RF': [0.5, -0.3, 0.8],
            'SMB': [0.1, 0.2, -0.1],
            'HML': [-0.2, 0.3, 0.1],
            'RMW': [0.1, -0.1, 0.2],
            'CMA': [0.05, 0.1, -0.05],
            'RF': [0.01, 0.02, 0.015]
        })
        result = dp._clean_ff(df)

        assert 'date' in result.columns
        assert 'Mkt-RF' in result.columns
        assert 'RF' not in result.columns or result['RF'].iloc[0] == pytest.approx(0.01 / 100.0)
        assert len(result) == 3

    def test_clean_ff_filters_invalid_dates(self):
        dp = DataProcessor()
        df = pd.DataFrame({
            'Unnamed: 0': ['202001', 'invalid', '202003', ''],
            'Mkt-RF': [0.5, -0.3, 0.8, 0.1],
            'SMB': [0.1, 0.2, -0.1, 0.0],
            'HML': [-0.2, 0.3, 0.1, 0.0],
            'RMW': [0.1, -0.1, 0.2, 0.0],
            'CMA': [0.05, 0.1, -0.05, 0.0],
            'RF': [0.01, 0.02, 0.015, 0.01]
        })
        result = dp._clean_ff(df)

        assert len(result) == 2  # Only valid dates

    def test_clean_ff_rf_scaling(self):
        dp = DataProcessor()
        df = pd.DataFrame({
            'Unnamed: 0': ['202001'],
            'Mkt-RF': [0.5],
            'SMB': [0.1],
            'HML': [-0.2],
            'RMW': [0.1],
            'CMA': [0.05],
            'RF': [1.0]  # 1% as percentage
        })
        result = dp._clean_ff(df)

        assert result['RF'].iloc[0] == pytest.approx(0.01, rel=1e-5)


class TestCleanAssetData:
    def test_clean_asset_data_fills_market_cols(self):
        dp = DataProcessor()
        dates = pd.date_range('2020-01-31', periods=3, freq='M')
        df = pd.DataFrame({
            'FIAT_price': [100.0, np.nan, 102.0],
            'FIAT_pct_ret': [0.01, np.nan, 0.02],
            'GOLD_price': [1500.0, 1510.0, np.nan],
            'GOLD_pct_ret': [0.01, 0.02, np.nan],
            'VDE_price': [50.0, 51.0, 52.0],
            'VDE_pct_ret': [0.01, 0.02, 0.03],
            'MSF_price': [3000.0, 3100.0, 3200.0],
            'MSF_pct_ret': [0.03, 0.04, 0.05],
            'BTC_pct_ret': [0.1, 0.2, 0.3],
            'ETH_pct_ret': [0.15, 0.25, 0.35],
            'LTC_pct_ret': [0.05, 0.1, 0.15],
            'BCH_pct_ret': [0.08, 0.12, 0.18],
        }, index=dates)

        result = dp._clean_asset_data(df)

        # Forward fill should work
        assert not result['FIAT_pct_ret'].isna().any()
        assert not result['GOLD_pct_ret'].isna().any()

    def test_clean_asset_data_requires_3_crypto(self):
        dp = DataProcessor()
        dates = pd.date_range('2020-01-31', periods=3, freq='M')
        df = pd.DataFrame({
            'FIAT_price': [100.0, 101.0, 102.0],
            'FIAT_pct_ret': [0.01, 0.01, 0.02],
            'GOLD_price': [1500.0, 1510.0, 1520.0],
            'GOLD_pct_ret': [0.01, 0.02, 0.01],
            'VDE_price': [50.0, 51.0, 52.0],
            'VDE_pct_ret': [0.01, 0.02, 0.03],
            'MSF_price': [3000.0, 3100.0, 3200.0],
            'MSF_pct_ret': [0.03, 0.04, 0.05],
            'BTC_pct_ret': [0.1, np.nan, 0.3],  # One has only 2 crypto available
            'ETH_pct_ret': [0.15, np.nan, 0.35],
            'LTC_pct_ret': [0.05, 0.1, 0.15],
            'BCH_pct_ret': [0.08, 0.12, 0.18],
        }, index=dates)

        result = dp._clean_asset_data(df)

        # Row with only 2 crypto should be dropped
        assert len(result) == 2


class TestMergeFF:
    def test_merge_ff_joins_correctly(self):
        dp = DataProcessor()

        dates = pd.date_range('2020-01-31', periods=3, freq='M')
        df = pd.DataFrame({
            'BTC_pct_ret': [0.1, 0.2, 0.3],
            'split': ['train', 'train', 'train']
        }, index=dates)

        ff = pd.DataFrame({
            'date': [pd.Period('2020-01', 'M'), pd.Period('2020-02', 'M'), pd.Period('2020-03', 'M')],
            'Mkt-RF': [0.5, -0.3, 0.8],
            'SMB': [0.1, 0.2, -0.1],
            'HML': [-0.2, 0.3, 0.1],
            'RMW': [0.1, -0.1, 0.2],
            'CMA': [0.05, 0.1, -0.05],
            'RF': [0.01, 0.02, 0.015]
        })

        result = dp._merge_ff(df, ff)

        assert 'Mkt-RF' in result.columns
        assert 'rf' in result.columns  # RF renamed to rf
        assert len(result) == 3


class TestMakeLaggedFeatures:
    def test_creates_lag_features(self):
        dp = DataProcessor()
        dates = pd.date_range('2020-01-31', periods=10, freq='M')
        df = pd.DataFrame({
            'crypto_ret': [0.1 * i for i in range(10)],
            'market_ret': [0.05 * i for i in range(10)],
            'split': ['train'] * 10
        }, index=dates)

        result, feat_cols = dp.make_lagged_features_for_model(
            df, 'crypto_ret', 'market_ret', w=3
        )

        # Check lag columns exist
        assert 'crypto_lag1' in feat_cols
        assert 'crypto_lag2' in feat_cols
        assert 'crypto_lag3' in feat_cols
        assert 'market_lag1' in feat_cols
        assert 'market_lag2' in feat_cols
        assert 'market_lag3' in feat_cols

    def test_creates_next_return_columns(self):
        dp = DataProcessor()
        dates = pd.date_range('2020-01-31', periods=10, freq='M')
        df = pd.DataFrame({
            'crypto_ret': [0.1 * i for i in range(10)],
            'market_ret': [0.05 * i for i in range(10)],
            'split': ['train'] * 10
        }, index=dates)

        result, feat_cols = dp.make_lagged_features_for_model(
            df, 'crypto_ret', 'market_ret', w=2
        )

        assert 'crypto_ret_next' in result.columns
        assert 'market_ret_next' in result.columns

    def test_drops_rows_with_missing_next_returns(self):
        dp = DataProcessor()
        dates = pd.date_range('2020-01-31', periods=5, freq='M')
        df = pd.DataFrame({
            'crypto_ret': [0.1, 0.2, 0.3, 0.4, 0.5],
            'market_ret': [0.05, 0.1, 0.15, 0.2, 0.25],
            'split': ['train'] * 5
        }, index=dates)

        result, feat_cols = dp.make_lagged_features_for_model(
            df, 'crypto_ret', 'market_ret', w=1
        )

        # Last row should be dropped since there's no "next" return
        assert len(result) == 4

    def test_fills_lag_na_with_zero(self):
        dp = DataProcessor()
        dates = pd.date_range('2020-01-31', periods=5, freq='M')
        df = pd.DataFrame({
            'crypto_ret': [0.1, 0.2, 0.3, 0.4, 0.5],
            'market_ret': [0.05, 0.1, 0.15, 0.2, 0.25],
            'split': ['train'] * 5
        }, index=dates)

        result, feat_cols = dp.make_lagged_features_for_model(
            df, 'crypto_ret', 'market_ret', w=2
        )

        # Check that feature columns don't have NaN
        for col in feat_cols:
            if col in result.columns:
                assert not result[col].isna().any(), f"Column {col} has NaN values"


class TestRunIntegration:
    @patch.object(DataProcessor, '_load_asset_data')
    @patch.object(DataProcessor, '_clean_asset_data')
    def test_run_without_ff(self, mock_clean, mock_load):
        dates = pd.date_range('2018-01-31', periods=12, freq='M')
        mock_df = pd.DataFrame({
            'BTC_pct_ret': [0.1] * 12
        }, index=dates)
        mock_load.return_value = mock_df
        mock_clean.return_value = mock_df

        dp = DataProcessor(with_ff=False)
        result = dp.run()

        mock_load.assert_called_once()
        mock_clean.assert_called_once()
        assert 'split' in result.columns

    @patch.object(DataProcessor, '_load_asset_data')
    @patch.object(DataProcessor, '_clean_asset_data')
    @patch.object(DataProcessor, '_load_ff')
    @patch.object(DataProcessor, '_clean_ff')
    @patch.object(DataProcessor, '_merge_ff')
    def test_run_with_ff(self, mock_merge, mock_clean_ff, mock_load_ff,
                         mock_clean, mock_load):
        dates = pd.date_range('2018-01-31', periods=12, freq='M')
        mock_df = pd.DataFrame({
            'BTC_pct_ret': [0.1] * 12
        }, index=dates)
        mock_load.return_value = mock_df
        mock_clean.return_value = mock_df
        mock_load_ff.return_value = pd.DataFrame()
        mock_clean_ff.return_value = pd.DataFrame()
        mock_merge.return_value = mock_df

        dp = DataProcessor(with_ff=True)
        result = dp.run()

        mock_load.assert_called_once()
        mock_clean.assert_called_once()
        mock_load_ff.assert_called_once()
        mock_clean_ff.assert_called_once()
        mock_merge.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
