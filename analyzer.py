import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional


class Analyzer:
    def __init__(self, with_ff: bool = False):
        self.with_ff = with_ff
        self.crypto_colors = {
            'BTC': '#f7931a',
            'ETH': '#627eea',
            'LTC': '#345d9d',
            'BCH': '#8dc351'
        }

    def plot_learning_curves(self, history: Dict[str, List[float]],
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> None:
        train = history.get("train", [])
        val = history.get("val", [])

        if not train and not val:
            print("No training history to plot.")
            return

        default_title = "Training Curves (RMSE)"
        if self.with_ff:
            default_title += " - With Fama-French Factors"
        else:
            default_title += " - Baseline"

        plot_title = title if title else default_title

        plt.figure(figsize=(10, 6))

        if train:
            plt.plot(range(1, len(train) + 1), train, label="Train", linewidth=2)
        if val:
            plt.plot(range(1, len(val) + 1), val, label="Validation", linewidth=2)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.title(plot_title, fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def compute_descriptive_statistics(self, test_long: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        rows = []

        for pair, g in test_long.groupby("pair"):
            v = g["beta_hat"].dropna()

            if len(v) == 0:
                continue

            rows.append({
                "Pair": pair,
                "N": len(v),
                "Mean": v.mean(),
                "Std Dev": v.std(),
                "Skewness": stats.skew(v),
                "Excess Kurtosis": stats.kurtosis(v),
                "Min": v.min(),
                "P1": v.quantile(0.01),
                "P5": v.quantile(0.05),
                "P25": v.quantile(0.25),
                "P50 (Median)": v.quantile(0.50),
                "P75": v.quantile(0.75),
                "P95": v.quantile(0.95),
                "P99": v.quantile(0.99),
                "Max": v.max(),
            })
        desc_stats = pd.DataFrame(rows).sort_values("Pair").reset_index(drop=True)
        if save_path:
            desc_stats.to_csv(save_path)
        return desc_stats

    def plot_neural_beta_dynamics(self, test_long: pd.DataFrame,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> pd.DataFrame:
        df = test_long.copy()
        if pd.api.types.is_period_dtype(df["date"]): # type: ignore
            df["date"] = df["date"].dt.to_timestamp() # type: ignore
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year # type: ignore

        annual_betas = (
            df
            .groupby(["year", "crypto", "asset"], as_index=False)
            .agg(
                mean_beta=("beta_hat", "mean"),
                std_beta=("beta_hat", "std"),
                n=("beta_hat", "count")
            )
        )

        assets = sorted(annual_betas["asset"].unique())
        cryptos = sorted(annual_betas["crypto"].unique())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, asset in enumerate(assets):
            ax = axes[idx]
            asset_df = annual_betas[annual_betas["asset"] == asset]

            for crypto in cryptos:
                cdf = asset_df[asset_df["crypto"] == crypto].sort_values("year")

                if len(cdf) > 0:
                    color = self.crypto_colors.get(crypto, None)
                    ax.plot(
                        cdf["year"],
                        cdf["mean_beta"],
                        marker="o",
                        label=crypto,
                        color=color,
                        linewidth=2,
                        markersize=8
                    )

                    ax.fill_between(
                        cdf["year"],
                        cdf["mean_beta"] - cdf["std_beta"],
                        cdf["mean_beta"] + cdf["std_beta"],
                        alpha=0.15,
                        color=color
                    )

            ax.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=1)
            ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, linewidth=1, label="beta=1")

            ax.set_title(f"Neural Beta: Crypto vs {asset.capitalize()}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Mean Neural Beta", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)

            if len(cdf) > 0:
                years = sorted(cdf["year"].unique())
                ax.set_xticks(years)
                ax.set_xticklabels(years)

        default_title = "Evolution of Neural Betas Over Time (Test Period)"
        if self.with_ff:
            default_title += " - With Fama-French Factors"
        else:
            default_title += " - Baseline"

        suptitle = title if title else default_title
        plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return annual_betas

    @staticmethod
    def compare_baseline_vs_ff(meta_baseline: pd.DataFrame,
                                meta_ff: pd.DataFrame,
                                save_path: Optional[str] = None) -> pd.DataFrame:
        comparison = pd.merge(
            meta_baseline[['crypto', 'asset', 'val_loss']].rename(columns={'val_loss': 'baseline_val_rmse'}),
            meta_ff[['crypto', 'asset', 'val_loss']].rename(columns={'val_loss': 'ff_val_rmse'}),
            on=['crypto', 'asset'],
            how='outer'
        )

        comparison['improvement'] = comparison['baseline_val_rmse'] - comparison['ff_val_rmse']
        comparison['pct_improvement'] = (comparison['improvement'] / comparison['baseline_val_rmse']) * 100

        print(f"\nMean improvement: {comparison['improvement'].mean():.6f}")
        print(f"Mean % improvement: {comparison['pct_improvement'].mean():.2f}%")
        print(f"Pairs improved: {(comparison['improvement'] > 0).sum()}/{len(comparison)}")
        print(f"Pairs worsened: {(comparison['improvement'] < 0).sum()}/{len(comparison)}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        x = np.arange(len(comparison))
        width = 0.35
        ax.bar(x - width/2, comparison['baseline_val_rmse'], width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, comparison['ff_val_rmse'], width, label='With FF', alpha=0.8)
        ax.set_xlabel('Crypto-Asset Pair')
        ax.set_ylabel('Validation RMSE')
        ax.set_title('RMSE: Baseline vs Fama-French')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row['crypto']}-{row['asset']}" for _, row in comparison.iterrows()],
            rotation=45, ha='right', fontsize=8
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1]
        colors = ['green' if val > 0 else 'red' for val in comparison['pct_improvement']]
        ax.barh(range(len(comparison)), comparison['pct_improvement'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(comparison)))
        ax.set_yticklabels(
            [f"{row['crypto']}-{row['asset']}" for _, row in comparison.iterrows()],
            fontsize=9
        )
        ax.set_xlabel('Improvement (%)')
        ax.set_title('% RMSE Improvement with FF Factors')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return comparison

    def create_portfolio_analysis(self, test_long: pd.DataFrame, df: pd.DataFrame,
                                  save_path: Optional[str] = None) -> tuple:
        asset_col_map = {
            'equity': 'MSF_pct_ret',
            'fiat': 'FIAT_pct_ret',
            'gold': 'GOLD_pct_ret',
            'energy': 'VDE_pct_ret'
        }

        portfolio_df = test_long.copy()
        if pd.api.types.is_period_dtype(portfolio_df["date"]): # type: ignore
            portfolio_df["date"] = portfolio_df["date"].dt.to_timestamp() # type: ignore
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])

        crypto_ret_next = []
        asset_ret_next = []

        for _, row in portfolio_df.iterrows():
            date = row['date']
            crypto = row['crypto']
            asset = row['asset']

            crypto_col = f'{crypto}_pct_ret'
            asset_col = asset_col_map.get(asset)

            next_date = date + pd.DateOffset(months=1)

            if next_date in df.index:
                crypto_ret_next.append(df.loc[next_date, crypto_col])
                asset_ret_next.append(df.loc[next_date, asset_col] if asset_col else np.nan)
            else:
                crypto_ret_next.append(np.nan)
                asset_ret_next.append(np.nan)

        portfolio_df['crypto_ret_next'] = crypto_ret_next
        portfolio_df['asset_ret_next'] = asset_ret_next
        portfolio_df = portfolio_df.dropna(subset=['crypto_ret_next', 'asset_ret_next'])

        crypto_abs_mean_ret = {}
        for crypto in portfolio_df['crypto'].unique():
            crypto_col = f'{crypto}_pct_ret'
            crypto_abs_mean_ret[crypto] = df[crypto_col].abs().mean()

        portfolio_df['vw_weight'] = portfolio_df['crypto'].map(crypto_abs_mean_ret)

        portfolio_df["year_month"] = portfolio_df["date"].dt.to_period("M") # type: ignore
        portfolio_df["excess_ret"] = portfolio_df["crypto_ret_next"] - portfolio_df["asset_ret_next"]

        portfolio_df["portfolio"] = (
            portfolio_df.groupby(["asset", "year_month"])["beta_hat"]
            .rank(method="first")
            .astype(int)
        )

        monthly_stats = []
        for (asset, ym, portfolio), grp in portfolio_df.groupby(["asset", "year_month", "portfolio"]):
            # Equal-weighted statistics
            ew_beta = grp["beta_hat"].mean()
            ew_excess = grp["excess_ret"].mean()

            # Value-weighted statistics (weights = absolute mean return of each crypto)
            weights = grp["vw_weight"]
            if weights.sum() > 0:
                vw_beta = np.average(grp["beta_hat"], weights=weights)
                vw_excess = np.average(grp["excess_ret"], weights=weights)
            else:
                vw_beta = np.nan
                vw_excess = np.nan

            monthly_stats.append({
                "asset": asset,
                "year_month": ym,
                "portfolio": portfolio,
                "n_cryptos": len(grp),
                "ew_beta": ew_beta,
                "ew_excess_ret": ew_excess,
                "vw_beta": vw_beta,
                "vw_excess_ret": vw_excess,
            })

        portfolio_monthly = pd.DataFrame(monthly_stats)

        portfolio_stats = (
            portfolio_monthly
            .groupby(["asset", "portfolio"])
            .agg({
                "n_cryptos": "mean",
                "ew_beta": "mean",
                "ew_excess_ret": "mean",
                "vw_beta": "mean",
                "vw_excess_ret": "mean",
            })
            .reset_index()
        )

        hml_rows = []
        for asset in portfolio_stats["asset"].unique():
            sub = portfolio_stats[portfolio_stats["asset"] == asset]

            high = sub[sub["portfolio"] == 4]
            low = sub[sub["portfolio"] == 1]

            if len(high) > 0 and len(low) > 0:
                hml_rows.append({
                    "Asset": asset,
                    "EW Beta Spread": high["ew_beta"].values[0] - low["ew_beta"].values[0],
                    "EW Excess Return Diff (%)": (high["ew_excess_ret"].values[0] - low["ew_excess_ret"].values[0]) * 100,
                    "VW Beta Spread": high["vw_beta"].values[0] - low["vw_beta"].values[0],
                    "VW Excess Return Diff (%)": (high["vw_excess_ret"].values[0] - low["vw_excess_ret"].values[0]) * 100,
                })

        hml_df = pd.DataFrame(hml_rows)

        portfolio_stats["ew_excess_ret"] = portfolio_stats["ew_excess_ret"] * 100
        portfolio_stats["vw_excess_ret"] = portfolio_stats["vw_excess_ret"] * 100

        portfolio_stats = portfolio_stats.rename(columns={
            "asset": "Asset",
            "portfolio": "Portfolio (Quartile)",
            "n_cryptos": "N Cryptos",
            "ew_beta": "EW Mean Beta",
            "ew_excess_ret": "EW Excess Return (%)",
            "vw_beta": "VW Mean Beta",
            "vw_excess_ret": "VW Excess Return (%)",
        })

        if save_path:
            portfolio_stats.to_csv(save_path, index=False)

        return portfolio_stats, hml_df
