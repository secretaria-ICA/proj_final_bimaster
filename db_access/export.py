import pandas as pd
import os
from abc import ABC, abstractmethod
import pickle


class AbstractExport(ABC):
    @abstractmethod
    def export(self, obj: object, folder_path: str, file_name: str) -> str:
        pass

class ExportToParquet(AbstractExport):
    def export(self, df: pd.DataFrame, folder_path: str, file_name: str) -> str:
        assert df.shape[0] > 0, "The DataFrame can not be empty."

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if not file_name.endswith(".parquet"):
            file_name = file_name + ".parquet"

        full_name = os.path.join(folder_path, file_name)
        if os.path.exists(full_name):
            os.remove(full_name)
            
        df.to_parquet(full_name)

        return full_name

class ExportToPickle(AbstractExport):
    def export(self, model: object, folder_path: str, file_name: str)->str:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if not file_name.endswith(".pickle"):
            file_name = file_name + ".pickle"

        full_name = os.path.join(folder_path, file_name)
        with open(full_name, "wb") as f:
            pickle.dump(model, f)

        return full_name