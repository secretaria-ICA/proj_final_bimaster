import os
import talib
import re
import pandas as pd
import json
import enum
import numpy as np
import pickle
from typing import List
from talib import abstract
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from db_access import ExportToParquet, ExportToPickle


class PROFTABILITY_TYPE(enum.IntEnum):
    LINEAR = 1
    LOG = 2

class PreProcess():
    
    def apply_function(self, data_source: pd.DataFrame, func_name: str, **kwargs)->pd.DataFrame:
        assert func_name in talib.get_functions(), f"Invalid function {func_name}"
        price_params = ["open", "close", "high", "low", "volume"]
        price_params_vals = []

        regex = re.compile(r"\{.*\}")

        # Create a copy of the input dataframe that will be used as the output dataframe
        df_ret = data_source.copy(deep=True)
        with open("func_defs.json", "r") as f:
            func_def = json.load(f)
            for type in func_def.keys():
                if func_name in func_def[type]:
                    input_params = func_def[type][func_name]['input_params']
                    # Validate the input parameter
                    for param in input_params:
                        if param not in price_params and param not in kwargs:
                            raise ValueError(f"The function {func_name} requires the parameter {param}.")
                        elif param in price_params:
                            price_params_vals.append(data_source[param])
                    
                    return_values = func_def[type][func_name]['return_values']
                    # Adjust the name of the output parameters
                    for i, ret_val in enumerate(return_values):
                        param_name = regex.search(ret_val)
                        if param_name is not None:
                            param_name = param_name.group(0)[1:-1]
                            return_values[i] = regex.sub(str(kwargs[param_name]), ret_val)

                    # Create areference to the function used to calculate the technical indicator
                    ta_lib_function = abstract.Function(func_name)

                    if len(return_values) == 1:
                        df_ret[return_values[0]] = ta_lib_function(*price_params_vals, **kwargs)
                    else:
                        ret_aux = []
                        ret_aux.extend(ta_lib_function(*price_params_vals, **kwargs))
                        for i, ret in enumerate(ret_aux):
                            df_ret[return_values[i]] = ret

                    return df_ret


    def calculate_strategy(self, strategy_file: str, data_set: pd.DataFrame)->pd.DataFrame:
        assert os.path.exists(strategy_file)
        # Read the strategy from the configuration file
        with open(strategy_file, "r") as f:
            startegies = json.load(f)
            for strategy in startegies.keys():
                df_ret = data_set.copy(deep=True)
                print(f"Processing strategy: {startegies[strategy]['description']}")
                for function in startegies[strategy]["functions"]:
                    dict_aux = startegies[strategy]["functions"]
                    print(f"Calculating {function}...")
                    if "params" in dict_aux[function]:
                        df_ret = self.apply_function(df_ret, dict_aux[function]['function'], **dict_aux[function]['params'])
                    else:
                        df_ret = self.apply_function(df_ret, dict_aux[function]['function'])

                yield strategy, startegies[strategy], df_ret


    def transpose_columns(self,
                          df: pd.DataFrame, 
                          window_size: int, 
                          shift: int = 1,
                          cols_to_transpose: List[str] = None,
                          dt_column: str = "dt_price",
                          ticker_column: str = "ticker")->pd.DataFrame:

        assert window_size <= df.shape[0], "The window size must be less than the number of rows"
        assert shift > 0, "The parameter shift must be an integer greater than zero"
        assert pd.api.types.is_datetime64_any_dtype(df[dt_column]), "The parameter dt_column must be a datetime64"

        # if no list is passed, set the lisof columns to transpose
        if cols_to_transpose is None:
            cols_to_transpose = [col for col in df.columns if col not in [ticker_column, dt_column]]

        col_names = [f"start_{dt_column}", f"end_{dt_column}"]
        for col in cols_to_transpose:
            # create the list of columns based on the window size
            col_names.extend([f"{col}_{i}" for i in range(window_size)])
        
        i = 0
        j = window_size
        values = []
        while j <= df.shape[0]:
            row_values = [df[dt_column].iloc[i], df[dt_column].iloc[j-1]]
            for col in cols_to_transpose:
                row_values.extend(df[col].iloc[i:j].values)
            
            values.append(row_values)

            i += shift
            j = i+window_size
        
        df_ret = pd.DataFrame(data=values, columns=col_names)
        df_ret.insert(0, ticker_column, [df[ticker_column].unique()[0]]*df_ret.shape[0])

        return df_ret


    def calculate_proftability(self, df: pd.DataFrame, 
                               dt_search: pd.Timestamp, 
                               profit_period: int, 
                               calc_type: PROFTABILITY_TYPE = PROFTABILITY_TYPE.LINEAR)->float:
        if df.index.get_loc(dt_search)+profit_period < df.shape[0]:
            return df.iloc[df.index.get_loc(dt_search)+profit_period]["close"]/df.iloc[df.index.get_loc(dt_search)+1]["close"]-1 \
                   if calc_type == PROFTABILITY_TYPE.LINEAR \
                   else np.log(df.iloc[df.index.get_loc(dt_search)+profit_period]["close"]/df.iloc[df.index.get_loc(dt_search)+1]["close"])
        else:
            return None

    
    def create_train_test_dataset(self, strategy_file: str, test_size: float, random_seed: int, scaler: TransformerMixin):
        data_file_path = os.environ.get("DATASET_PATH")
        model_base_path = os.environ.get("MODELS_PATH")
        train_ds_base_path = os.environ.get("TRAIN_DATASET")
        price_cols_to_delete = ["open", "high", "low"]
        exp_to_parquet = ExportToParquet()

        with open(strategy_file, "r") as f:
            df_train = None
            df_test = None
            startegies = json.load(f)
            for strategy in startegies.keys():
                files_path = os.path.join(data_file_path, strategy)
                path_content = os.listdir(files_path)
                # Filtra os arquivos parquet do diretório
                path_content = [file for file in path_content if file.endswith(".parquet")]

                for file in path_content:
                    print(f"Processando arquivo {file} na estrategia {strategy}")
                    asset = file.split('.')[0]
                    cols_to_delete = ["ticker", "start_dt_price", "end_dt_price", "proftability"]
                    df = pd.read_parquet(os.path.join(files_path, file))
                    # Seleciona as colunas que nao serao usadas no modelo
                    for price_col in price_cols_to_delete:
                        for df_col in df.columns:
                            if df_col.startswith(price_col):
                                cols_to_delete.append(df_col)

                    df.drop(columns=cols_to_delete, inplace=True)
                    Y = df.pop('label')
                    remaining_cols_df = df.columns

                    if df_test is None:
                        df_test = pd.DataFrame(data=None, columns=remaining_cols_df)
                        df_train = pd.DataFrame(data=None, columns=remaining_cols_df)

                    # Gera as bases de treino e teste
                    X_train, X_test, Y_train, Y_test = train_test_split(df.values, 
                                                                        Y.values, 
                                                                        test_size=test_size, 
                                                                        stratify=Y.values, 
                                                                        random_state=random_seed)

                    scalar_instance = scaler.fit(X_train)

                    # salva o modelo que faz scaling para ser usado posteriormente
                    exp_to_pickle = ExportToPickle()
                    exp_to_pickle.export(scalar_instance, os.path.join(model_base_path, strategy), asset)

                    for aux_vals, aux_labels, suffix in [(X_train, Y_train, "train"), (X_test, Y_test, "test")]:
                        X_standarized = scalar_instance.transform(aux_vals)
                        df_aux = pd.DataFrame(data=X_standarized, columns=remaining_cols_df)
                        df_aux["label"] = aux_labels
                        if suffix == "train":
                            df_train = pd.concat([df_train, df_aux], ignore_index=True)
                        else:
                            df_test = pd.concat([df_test, df_aux], ignore_index=True)

                exp_to_parquet.export(df_train, os.path.join(train_ds_base_path, strategy), "train_data")
                exp_to_parquet.export(df_test, os.path.join(train_ds_base_path, strategy), "test_data")
