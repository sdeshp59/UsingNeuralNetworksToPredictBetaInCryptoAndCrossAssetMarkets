from data_processor import *
from mlp import *
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

SEED = 20030910
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FineTune():
    def __init__(self
                ,crypto_list=['BTC', 'ETH', 'LTC', 'BCH']
                ,market_combinations=[('MSF_pct_ret', 'equity'),('FIAT_pct_ret', 'fiat'),('GOLD_pct_ret', 'gold'),('VDE_pct_ret', 'energy')]
                ,w_list = (12, 24, 36)
                ,hidden_list = (4,8,16)
                ,lr_list = (1e-4, 1e-3, 1e-2)
                ,activations = ("linear", "sigmoid", "tanh", "relu")
                ,batch_size=16
                ,max_epochs=20
                ,with_ff = False):
        
        self.processor = DataProcessor()
        self.cryptos = crypto_list
        self.mkt_combos = market_combinations
        self.w_list = w_list
        self.hidden_list = hidden_list
        self.lr_list = lr_list
        self.activations = activations
        self.batch_size=batch_size
        self.max_epochs=max_epochs
        self.with_ff = with_ff
        
    def grid_search_mlp(self, df):
        results = []
        best = {
            "val": float("inf"),
            "combo": None,
            "state": None,
            "history": None,
            "in_dim": None,
            "scaler": None
        }

        for crypto in self.cryptos:
            crypto_col = f'{crypto}_pct_ret'

            if crypto_col not in df.columns:
                continue

            for market_col, market_name in self.mkt_combos:
                for w in self.w_list:
                    temp_df, feat_cols = self.processor.make_lagged_features_for_model(
                        df, crypto_col, market_col, w,
                        with_ff=self.with_ff
                    )

                    train_df = temp_df[temp_df['split'] == 'train']
                    val_df = temp_df[temp_df['split'] == 'val']

                    if len(train_df) < 10 or len(val_df) < 5:
                        continue
                    
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(train_df[feat_cols].values)
                    X_val = scaler.transform(val_df[feat_cols].values)

                    r_train = train_df['crypto_ret_next'].values
                    m_train = train_df['market_ret_next'].values
                    r_val = val_df['crypto_ret_next'].values
                    m_val = val_df['market_ret_next'].values

                    ds_train = NeuralBetaDataset(X_train, r_train, m_train)
                    ds_val = NeuralBetaDataset(X_val, r_val, m_val)

                    for hidden_dim in self.hidden_list:
                        for lr in self.lr_list:
                            for act in self.activations:
                                try:
                                    model = NeuralBetaMLP(
                                        in_dim=len(feat_cols),
                                        hidden_dim=hidden_dim,
                                        activation=act
                                    )
                                    trainer = NeuralBetaTrainer(model)
                                    out = trainer.fit(ds_train, ds_val)

                                    val_loss = out["best_val"]

                                    results.append({
                                        "crypto": crypto,
                                        "market": market_name,
                                        "w": w,
                                        "hidden": hidden_dim,
                                        "lr": lr,
                                        "activation": act,
                                        "val_loss": val_loss
                                    })

                                    if val_loss < best["val"]:
                                        best = {
                                            "val": val_loss,
                                            "combo": {
                                                "crypto": crypto,
                                                "market": market_name,
                                                "market_col": market_col,
                                                "w": w,
                                                "hidden": hidden_dim,
                                                "lr": lr,
                                                "activation": act
                                            },
                                            "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                                            "in_dim": len(feat_cols),
                                            "history": out["history"],
                                            "scaler": scaler
                                        }

                                except Exception as e:
                                    print(f"ERROR: {e}")
                                    continue
        return results, best

    def generate_test_predictions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_test_outputs = []
        all_best_meta = []

        for crypto in self.cryptos:
            crypto_col = f'{crypto}_pct_ret'
            if crypto_col not in df.columns:
                continue

            for market_col, market_name in self.mkt_combos:
                pair_tuner = FineTune(
                    crypto_list=[crypto],
                    market_combinations=[(market_col, market_name)],
                    w_list=self.w_list,
                    hidden_list=self.hidden_list,
                    lr_list=self.lr_list,
                    activations=self.activations,
                    batch_size=self.batch_size,
                    max_epochs=self.max_epochs,
                    with_ff=self.with_ff
                )
                _, best = pair_tuner.grid_search_mlp(df)

                if best["combo"] is None:
                    continue

                combo = best["combo"]
                w = combo["w"]

                temp_df, feat_cols = self.processor.make_lagged_features_for_model(
                    df, crypto_col, market_col, w, with_ff=self.with_ff
                )
                test_df = temp_df[temp_df['split'] == 'test'].copy()

                if len(test_df) == 0:
                    continue

                best_model = NeuralBetaMLP(
                    in_dim=best["in_dim"],
                    hidden_dim=combo["hidden"],
                    activation=combo["activation"]
                ).to(DEVICE)
                best_model.load_state_dict(best["state"])
                best_model.eval()

                X_test = best["scaler"].transform(test_df[feat_cols].values)
                ds_test = NeuralBetaDataset(
                    X_test,
                    test_df['crypto_ret_next'].values,
                    test_df['market_ret_next'].values
                )
                test_loader = DataLoader(ds_test, batch_size=1024, shuffle=False)

                preds = []
                with torch.no_grad():
                    for batch in test_loader:
                        x = batch["x"].to(DEVICE)
                        beta_hat = best_model(x)
                        preds.append(beta_hat.cpu().numpy())

                out = test_df.copy()
                out["beta_hat"] = np.concatenate(preds)
                out["date"] = out.index
                out["crypto"] = crypto
                out["asset"] = market_name
                out["pair"] = f"{crypto}-{market_name}"

                all_test_outputs.append(out[["date", "crypto", "asset", "pair", "beta_hat"]])
                all_best_meta.append({
                    "crypto": crypto,
                    "asset": market_name,
                    "val_loss": best["val"],
                    "w": w,
                    "hidden": combo["hidden"],
                    "lr": combo["lr"],
                    "activation": combo["activation"]
                })

        test_long = pd.concat(all_test_outputs, ignore_index=True)
        best_meta_df = pd.DataFrame(all_best_meta)
        return test_long, best_meta_df