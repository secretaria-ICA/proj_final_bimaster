import os
import talib
import re
import pandas as pd
import json
import enum
import numpy as np
from typing import List
from talib import abstract
from datetime import timedelta


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