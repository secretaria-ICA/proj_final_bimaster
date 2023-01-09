import os
import talib
import re
import pandas as pd
import json
import enum
import math
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List
from talib import abstract
from sklearn.model_selection import train_test_split
from db_access import ExportToParquet


class PROFTABILITY_TYPE(enum.IntEnum):
    LINEAR = 1
    LOG = 2


class PivotLevels:
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series) -> None:
        self._low = low
        self._high = high
        self._close = close
        self._pp = (high + low + close)/3


    @property
    def R1(self)->pd.Series:
        return 2 * self._pp - self._low

    @property
    def S1(self)->pd.Series:
        return 2 * self._pp - self._high

    @property
    def R2(self)->pd.Series:
        return self._pp + (self._high - self._low)

    @property
    def S2(self)->pd.Series:
        return self._pp - (self._high - self._low)

    @property
    def R3(self)->pd.Series:
        return self._pp + 2 * (self._high - self._low)

    @property
    def S3(self)->pd.Series:
        return self._pp - 2 * (self._high - self._low)


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

                if "candles" in startegies[strategy]:
                    for candle_func in startegies[strategy]["candles"]:
                         # Create areference to the function used to calculate the technical indicator
                        ta_lib_function = abstract.Function(candle_func)
                        df_ret[candle_func] = ta_lib_function(df_ret["open"], df_ret["high"], df_ret["low"], df_ret["close"])/100

                if "custom_columns" in startegies[strategy]:
                    reg_exp_cols = re.compile(r"\[(.*?)\]")
                    reg_exp_ops = re.compile(r"\](.*?)\[")
                    for key in startegies[strategy]['custom_columns']:
                        cust_col = startegies[strategy]['custom_columns'][key]
                        ops = reg_exp_ops.findall(cust_col)
                        for op in ops:
                            op = op.strip()
                            assert op in ['-', '+', '*', '/'], f"Invalid operation[{op}]! The possible operations are +, -, * and /"

                        columns = reg_exp_cols.findall(cust_col)

                        for col in columns:
                            cust_col = cust_col.replace(f"[{col}]", f"df_ret['{col}']")     
                        
                        df_ret[key] = eval(cust_col)                    


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


    def format_dataset(self,
                       df_raw: pd.DataFrame,
                       df_tech: pd.DataFrame,
                       window_size: int,
                       stride: int,
                       profit_period: int,
                       min_profit: float,
                       cols_to_delete: List,
                       signal_cols: List,
                       ticker_col: str = "ticker",
                       date_col: str = "dt_price") -> pd.DataFrame:

        if cols_to_delete is not None:
            cols_to_delete.extend([ticker_col, date_col])
        else:
            cols_to_delete = [ticker_col, date_col]

        cols = [col for col in df_tech.columns if col not in cols_to_delete and col not in signal_cols]

        i = 0
        j = i + window_size
        rows = []

        while j < df_tech.shape[0]:
            profit = self.calculate_proftability(df_raw, df_tech[date_col].iloc[j], profit_period, PROFTABILITY_TYPE.LINEAR)
            rows.append([df_tech[ticker_col].iloc[i], 
                         df_tech[date_col].iloc[i], 
                         df_tech[date_col].iloc[j],
                         df_tech[cols].iloc[i:j].shape,
                         df_tech[cols].iloc[i:j].values.flatten(),
                         *df_tech[signal_cols].iloc[j].values,
                         profit,
                         int(profit >= min_profit)if profit else None])

            i += stride
            j = i + window_size

        df_col_names = [ticker_col, f"{date_col}_start", f"{date_col}_ends", "shape", "series", *signal_cols, "profit", "label"]
        df_ret = pd.DataFrame(rows, columns=df_col_names)
        df_ret.dropna(inplace=True)
        df_ret["label"] = df_ret["label"].astype(int)

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

    
    def create_train_test_dataset(self, strategy_file: str, test_size: float, random_seed: int):
        data_file_path = os.environ.get("DATASET_PATH")
        train_ds_base_path = os.environ.get("TRAIN_DATASET")
        price_cols_to_delete = ["open", "high", "low"]
        exporter = ExportToParquet()

        with open(strategy_file, "r") as f:
            df_train = None
            df_test = None
            startegies = json.load(f)
            for strategy in startegies.keys():
                files_path = os.path.join(data_file_path, strategy)
                path_content = os.listdir(files_path)
                # Filtra os arquivos parquet do diretório
                path_content = [file for file in path_content if file.endswith(".parquet")]
                new_dataframe_instance = True

                for file in path_content:
                    print(f"Processando arquivo {file} na estrategia {strategy}")
                    cols_to_delete = ["ticker", "dt_price_start", "dt_price_ends", "profit"]
                    df = pd.read_parquet(os.path.join(files_path, file))
                    # Seleciona as colunas que nao serao usadas no modelo
                    for price_col in price_cols_to_delete:
                        for df_col in df.columns:
                            if df_col.startswith(price_col):
                                cols_to_delete.append(df_col)

                    df.drop(columns=cols_to_delete, inplace=True)
                    Y = df.pop('label')
                    remaining_cols_df = df.columns

                    if new_dataframe_instance:
                        df_test = pd.DataFrame(data=None, columns=remaining_cols_df)
                        df_train = pd.DataFrame(data=None, columns=remaining_cols_df)
                        new_dataframe_instance = False

                    # Gera as bases de treino e teste
                    X_train, X_test, Y_train, Y_test = train_test_split(df.values, 
                                                                        Y.values, 
                                                                        test_size=test_size, 
                                                                        stratify=Y.values, 
                                                                        random_state=random_seed)

                    for aux_vals, aux_labels, suffix in [(X_train, Y_train, "train"), (X_test, Y_test, "test")]:
                        df_aux = pd.DataFrame(data=aux_vals, columns=remaining_cols_df)
                        df_aux["label"] = aux_labels
                        if suffix == "train":
                            df_train = pd.concat([df_train, df_aux], ignore_index=True)
                        else:
                            df_test = pd.concat([df_test, df_aux], ignore_index=True)

                exporter.export(df_train, os.path.join(train_ds_base_path, strategy), "train_data")
                exporter.export(df_test, os.path.join(train_ds_base_path, strategy), "test_data")


    def read_dataset_from_parquet(self, path: str)->np.array:
        assert os.path.exists(path), f"The file {path} does not exists!"
        df = pd.read_parquet(path)
        shape = df['shape'][0]

        df.series = df.series.transform(lambda val: val.reshape(shape))
        
        return df.drop(columns=['shape'])


    def linear_regression_slope(self, column: pd.Series, window_size: int, stride: int)->np.array:
        assert window_size > 0, "The parameter window size must be greater than 0"
        length = column.shape[0]
        array_size = math.ceil(column.shape[0]/stride)
        slopes = np.full(array_size, np.NaN)
        slope_index = slopes.shape[0] - 1
        for index in range(length, 0, -stride):
            if index >= window_size:
                X = np.array(range(window_size)).reshape(-1,1)
                Y = column[index-window_size:index].values
                model = LinearRegression().fit(X, Y)
                slopes[slope_index] = model.coef_
                slope_index -= 1
            else:
                break

        return slopes