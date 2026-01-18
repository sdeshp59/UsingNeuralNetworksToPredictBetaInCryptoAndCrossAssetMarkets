from data_processor import DataProcessor
from fine_tune import FineTune
from analyzer import Analyzer

import numpy as np
import pandas as pd
import random
import torch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

SEED = 20030910
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(with_ff: bool = False):
    suffix = '_ff' if with_ff else ''

    processor = DataProcessor(with_ff=with_ff)
    df = processor.run()

    fine_tuner = FineTune(with_ff=with_ff)

    results, best = fine_tuner.grid_search_mlp(df=df)

    print("\nBest overall configuration:")
    for key, value in best['combo'].items():
        print(f"  {key}: {value}")
    print(f"  Validation RMSE: {best['val']:.6f}")

    test_long, best_meta = fine_tuner.generate_test_predictions(df)

    analyzer = Analyzer(with_ff=with_ff)

    analyzer.plot_learning_curves(
        best['history'],
        save_path=f'outputs/learning_curves{suffix}.png'
    )

    desc_stats = analyzer.compute_descriptive_statistics(
        test_long,
        save_path=f'outputs/descriptive_stats{suffix}.csv'
    )
    
    annual_betas = analyzer.plot_neural_beta_dynamics(
        test_long,
        save_path=f'outputs/neural_beta_dynamics{suffix}.png'
    )

    portfolio_stats, hml_df = analyzer.create_portfolio_analysis(
        test_long,
        df,
        save_path=f'outputs/portfolio_stats{suffix}.csv'
    )

    print("\nPortfolio Statistics:")
    print(portfolio_stats.to_string(index=False))

    print("\nHigh-Minus-Low (HML) Spread:")
    print(hml_df.to_string(index=False))

    return test_long, best_meta, best, df


if __name__ == '__main__':
    test_long_no_ff, meta_no_ff, best_no_ff, df_no_ff = main(with_ff=False)
    test_long_ff, meta_ff, best_ff, df_ff = main(with_ff=True)

    print(f"\nWithout FF factors - Best val loss: {best_no_ff['val']:.6f}")
    print(f"With FF factors    - Best val loss: {best_ff['val']:.6f}")

    comparison = Analyzer.compare_baseline_vs_ff(
        meta_no_ff,
        meta_ff,
        save_path='outputs/ff_comparison.png'
    )
