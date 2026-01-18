import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from sklearn.preprocessing import StandardScaler

from fine_tune import FineTune
from data_processor import DataProcessor
from mlp import NeuralBetaDataset, NeuralBetaMLP, NeuralBetaTrainer


class TestFineTuneInit:
    def test_default_values(self):
        ft = FineTune()

        assert ft.cryptos == ['BTC', 'ETH', 'LTC', 'BCH']
        assert ft.mkt_combos == [
            ('MSF_pct_ret', 'equity'),
            ('FIAT_pct_ret', 'fiat'),
            ('GOLD_pct_ret', 'gold'),
            ('VDE_pct_ret', 'energy')
        ]
        assert ft.w_list == (12, 24, 36)
        assert ft.hidden_list == (4, 8, 16)
        assert ft.lr_list == (1e-4, 1e-3, 1e-2)
        assert ft.activations == ("linear", "sigmoid", "tanh", "relu")
        assert ft.batch_size == 16
        assert ft.max_epochs == 20
        assert ft.with_ff == False

    def test_custom_crypto_list(self):
        ft = FineTune(crypto_list=['BTC', 'ETH'])

        assert ft.cryptos == ['BTC', 'ETH']

    def test_custom_market_combinations(self):
        custom_combos = [('MSF_pct_ret', 'equity')]
        ft = FineTune(market_combinations=custom_combos)

        assert ft.mkt_combos == custom_combos

    def test_custom_w_list(self):
        ft = FineTune(w_list=(6, 12))

        assert ft.w_list == (6, 12)

    def test_custom_hidden_list(self):
        ft = FineTune(hidden_list=(8, 16, 32))

        assert ft.hidden_list == (8, 16, 32)

    def test_custom_lr_list(self):
        ft = FineTune(lr_list=(1e-5, 1e-4))

        assert ft.lr_list == (1e-5, 1e-4)

    def test_custom_activations(self):
        ft = FineTune(activations=("relu", "tanh"))

        assert ft.activations == ("relu", "tanh")

    def test_custom_batch_size(self):
        ft = FineTune(batch_size=32)

        assert ft.batch_size == 32

    def test_custom_max_epochs(self):
        ft = FineTune(max_epochs=50)

        assert ft.max_epochs == 50

    def test_with_ff_enabled(self):
        ft = FineTune(with_ff=True)

        assert ft.with_ff == True

    def test_processor_initialized(self):
        ft = FineTune()

        assert isinstance(ft.processor, DataProcessor)


class TestFineTuneGridSearch:
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range('2015-01-01', periods=n_samples, freq='ME')

        df = pd.DataFrame({
            'BTC_pct_ret': np.random.randn(n_samples) * 0.1,
            'ETH_pct_ret': np.random.randn(n_samples) * 0.1,
            'MSF_pct_ret': np.random.randn(n_samples) * 0.05,
            'FIAT_pct_ret': np.random.randn(n_samples) * 0.02,
        }, index=dates)

        # Add split column
        df['split'] = 'train'
        df.loc[df.index >= '2020-01-01', 'split'] = 'val'
        df.loc[df.index >= '2022-01-01', 'split'] = 'test'

        return df

    @pytest.fixture
    def minimal_fine_tune(self):
        """Create a FineTune instance with minimal hyperparameter grid."""
        return FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            batch_size=16,
            max_epochs=2
        )

    def test_grid_search_returns_tuple(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        assert isinstance(results, list)
        assert isinstance(best, dict)

    def test_grid_search_results_structure(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        if len(results) > 0:
            result = results[0]
            assert "crypto" in result
            assert "market" in result
            assert "w" in result
            assert "hidden" in result
            assert "lr" in result
            assert "activation" in result
            assert "val_loss" in result

    def test_grid_search_best_structure(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        assert "val" in best
        assert "combo" in best
        assert "state" in best
        assert "history" in best
        assert "in_dim" in best
        assert "scaler" in best

    def test_grid_search_best_combo_structure(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        if best["combo"] is not None:
            combo = best["combo"]
            assert "crypto" in combo
            assert "market" in combo
            assert "market_col" in combo
            assert "w" in combo
            assert "hidden" in combo
            assert "lr" in combo
            assert "activation" in combo

    def test_grid_search_scaler_is_fitted(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        if best["scaler"] is not None:
            assert isinstance(best["scaler"], StandardScaler)
            # Check that scaler has been fitted (has mean_ attribute)
            assert hasattr(best["scaler"], 'mean_')

    def test_grid_search_state_is_dict(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        if best["state"] is not None:
            assert isinstance(best["state"], dict)
            # Check that state contains tensor values
            for key, value in best["state"].items():
                assert isinstance(value, torch.Tensor)

    def test_grid_search_history_structure(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        if best["history"] is not None:
            assert "train" in best["history"]
            assert "val" in best["history"]
            assert "lr" in best["history"]

    def test_grid_search_skips_missing_crypto(self, sample_df):
        ft = FineTune(
            crypto_list=['XRP'],  # Not in sample_df
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        results, best = ft.grid_search_mlp(sample_df)

        assert len(results) == 0
        assert best["val"] == float("inf")
        assert best["combo"] is None

    def test_grid_search_multiple_cryptos(self, sample_df):
        ft = FineTune(
            crypto_list=['BTC', 'ETH'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        results, best = ft.grid_search_mlp(sample_df)

        cryptos_in_results = set(r["crypto"] for r in results)
        assert 'BTC' in cryptos_in_results or 'ETH' in cryptos_in_results

    def test_grid_search_multiple_markets(self, sample_df):
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity'), ('FIAT_pct_ret', 'fiat')],
            w_list=(6,),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        results, best = ft.grid_search_mlp(sample_df)

        markets_in_results = set(r["market"] for r in results)
        assert len(markets_in_results) >= 1

    def test_grid_search_val_loss_is_float(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        for result in results:
            assert isinstance(result["val_loss"], float)

    def test_grid_search_best_val_is_minimum(self, sample_df):
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4, 8),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        results, best = ft.grid_search_mlp(sample_df)

        if len(results) > 0:
            min_val_loss = min(r["val_loss"] for r in results)
            assert best["val"] == pytest.approx(min_val_loss, rel=1e-5)

    def test_grid_search_in_dim_matches_features(self, minimal_fine_tune, sample_df):
        results, best = minimal_fine_tune.grid_search_mlp(sample_df)

        if best["in_dim"] is not None:
            # With w=6, we have 6 crypto lags + 6 market lags = 12 features
            expected_dim = 6 * 2  # w * 2 (crypto + market)
            assert best["in_dim"] == expected_dim


class TestFineTuneEdgeCases:
    def test_empty_dataframe(self):
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        empty_df = pd.DataFrame()
        results, best = ft.grid_search_mlp(empty_df)

        assert len(results) == 0
        assert best["val"] == float("inf")

    def test_insufficient_train_data(self):
        """Test behavior when there's not enough training data."""
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        # Create a small DataFrame with insufficient data
        small_df = pd.DataFrame({
            'BTC_pct_ret': [0.1, 0.2, 0.3],
            'MSF_pct_ret': [0.01, 0.02, 0.03],
            'split': ['train', 'train', 'val']
        }, index=pd.date_range('2015-01-01', periods=3, freq='ME'))

        results, best = ft.grid_search_mlp(small_df)

        # Should skip due to insufficient data
        assert len(results) == 0

    def test_insufficient_val_data(self):
        """Test behavior when there's not enough validation data."""
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        # Create DataFrame with enough train but insufficient val
        np.random.seed(42)
        df = pd.DataFrame({
            'BTC_pct_ret': np.random.randn(50) * 0.1,
            'MSF_pct_ret': np.random.randn(50) * 0.05,
            'split': ['train'] * 48 + ['val'] * 2
        }, index=pd.date_range('2015-01-01', periods=50, freq='ME'))

        results, best = ft.grid_search_mlp(df)

        # Should skip due to insufficient validation data
        assert len(results) == 0


class TestFineTuneWithFamaFrench:
    @pytest.fixture
    def sample_df_with_ff(self):
        """Create a sample DataFrame with Fama-French factors."""
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range('2015-01-01', periods=n_samples, freq='ME')

        df = pd.DataFrame({
            'BTC_pct_ret': np.random.randn(n_samples) * 0.1,
            'MSF_pct_ret': np.random.randn(n_samples) * 0.05,
            'Mkt-RF': np.random.randn(n_samples) * 0.03,
            'SMB': np.random.randn(n_samples) * 0.02,
            'HML': np.random.randn(n_samples) * 0.02,
            'RMW': np.random.randn(n_samples) * 0.02,
            'CMA': np.random.randn(n_samples) * 0.02,
            'rf': np.random.randn(n_samples) * 0.001,
        }, index=dates)

        df['split'] = 'train'
        df.loc[df.index >= '2020-01-01', 'split'] = 'val'
        df.loc[df.index >= '2022-01-01', 'split'] = 'test'

        return df

    def test_grid_search_with_ff_factors(self, sample_df_with_ff):
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2,
            with_ff=True
        )

        results, best = ft.grid_search_mlp(sample_df_with_ff)

        # With FF factors, in_dim should be larger
        if best["in_dim"] is not None:
            # w=6 * (2 base + 6 FF factors) = 48 features
            expected_dim = 6 * (2 + 6)
            assert best["in_dim"] == expected_dim


class TestFineTuneActivations:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range('2015-01-01', periods=n_samples, freq='ME')

        df = pd.DataFrame({
            'BTC_pct_ret': np.random.randn(n_samples) * 0.1,
            'MSF_pct_ret': np.random.randn(n_samples) * 0.05,
        }, index=dates)

        df['split'] = 'train'
        df.loc[df.index >= '2020-01-01', 'split'] = 'val'

        return df

    def test_all_activations_work(self, sample_df):
        """Test that all activation functions work correctly."""
        for activation in ("linear", "sigmoid", "tanh", "relu"):
            ft = FineTune(
                crypto_list=['BTC'],
                market_combinations=[('MSF_pct_ret', 'equity')],
                w_list=(6,),
                hidden_list=(4,),
                lr_list=(1e-3,),
                activations=(activation,),
                max_epochs=2
            )

            results, best = ft.grid_search_mlp(sample_df)

            if len(results) > 0:
                assert results[0]["activation"] == activation


class TestFineTuneHyperparameterGrid:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range('2015-01-01', periods=n_samples, freq='ME')

        df = pd.DataFrame({
            'BTC_pct_ret': np.random.randn(n_samples) * 0.1,
            'MSF_pct_ret': np.random.randn(n_samples) * 0.05,
        }, index=dates)

        df['split'] = 'train'
        df.loc[df.index >= '2020-01-01', 'split'] = 'val'

        return df

    def test_grid_search_explores_all_hidden_dims(self, sample_df):
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4, 8),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        results, best = ft.grid_search_mlp(sample_df)

        hidden_dims = set(r["hidden"] for r in results)
        assert 4 in hidden_dims
        assert 8 in hidden_dims

    def test_grid_search_explores_all_window_sizes(self, sample_df):
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6, 12),
            hidden_list=(4,),
            lr_list=(1e-3,),
            activations=("relu",),
            max_epochs=2
        )

        results, best = ft.grid_search_mlp(sample_df)

        windows = set(r["w"] for r in results)
        assert 6 in windows
        assert 12 in windows

    def test_grid_search_result_count(self, sample_df):
        """Test that the number of results matches expected grid size."""
        ft = FineTune(
            crypto_list=['BTC'],
            market_combinations=[('MSF_pct_ret', 'equity')],
            w_list=(6,),
            hidden_list=(4, 8),
            lr_list=(1e-3, 1e-2),
            activations=("relu", "tanh"),
            max_epochs=2
        )

        results, best = ft.grid_search_mlp(sample_df)

        # Expected: 1 crypto * 1 market * 1 window * 2 hidden * 2 lr * 2 act = 8
        expected_count = 1 * 1 * 1 * 2 * 2 * 2
        assert len(results) == expected_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
