from typing import Dict

import numpy as np
import pandas as pd


class CryptoStore:
    def __init__(self) -> None:
        self.num_asset = 14
        self.length = 100
        self.asset_data = {
            i: {
                "count": np.zeros(self.length),
                "open": np.zeros(self.length),
                "high": np.zeros(self.length),
                "low": np.zeros(self.length),
                "close": np.zeros(self.length),
                "volume": np.zeros(self.length),
            }
            for i in range(self.num_asset)
        }

    def update(self, data: pd.DataFrame):
        for asset_id in range(self.num_asset):
            batch = data[data["Asset_ID"] == asset_id]

            batch_size = batch.shape[0]
            for col in ["count", "open", "high", "low", "close", "volume"]:
                pre_data = self.asset_data[asset_id][col].copy()
                self.asset_data[asset_id][col] = np.concatenate(
                    (pre_data[batch_size:], batch[col.title()])
                )
                assert (
                    self.asset_data[asset_id][col].shape[0] == self.length
                ), "Updated store must be store length."

    def get_asset_data(self, asset_id: int) -> Dict:
        return self.asset_data[asset_id]
