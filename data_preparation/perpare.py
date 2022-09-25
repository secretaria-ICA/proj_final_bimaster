import os
import talib
import re
import pandas as pd
import json
from talib import abstract
from db_access import ExportToParquet


class PreProcess():

    def __init__(self, strategy_file: str) -> None:
        assert os.path.exists(strategy_file)
        self.strategy_file = strategy_file
        
        
    def apply_function(self, data_source: pd.DataFrame, func_name: str, **kwargs)->pd.DataFrame:
        assert func_name in talib.get_functions(), f"Invalid function {func_name}"
        price_params = ["open", "close", "high", "low", "volume"]
        price_params_vals = []

        regex = re.compile(r"\{.*\}")

        # Cria uma copia do dataframe de entreda que sera usado como dataframe de saida
        df_ret = data_source.copy(deep=True)
        with open("func_defs.json", "r") as f:
            func_def = json.load(f)
            for type in func_def.keys():
                if func_name in func_def[type]:
                    input_params = func_def[type][func_name]['input_params']
                    # Valida os parametros de entrada da funcao
                    for param in input_params:
                        if param not in price_params and param not in kwargs:
                            raise ValueError(f"The function {func_name} requires the parameter {param}.")
                        elif param in price_params:
                            price_params_vals.append(data_source[param])
                    
                    return_values = func_def[type][func_name]['return_values']
                    # Ajusta o nome dos parametros de retorno
                    for i, ret_val in enumerate(return_values):
                        param_name = regex.search(ret_val)
                        if param_name is not None:
                            param_name = param_name.group(0)[1:-1]
                            return_values[i] = regex.sub(str(kwargs[param_name]), ret_val)

                    # Cria uma instancia da funcao para calcular o indicador tecnico
                    ta_lib_function = abstract.Function(func_name)

                    if len(return_values) == 1:
                        df_ret[return_values[0]] = ta_lib_function(*price_params_vals, **kwargs)
                    else:
                        ret_aux = []
                        ret_aux.extend(ta_lib_function(*price_params_vals, **kwargs))
                        for i, ret in enumerate(ret_aux):
                            df_ret[return_values[i]] = ret

                    return df_ret

    def create_datasets(self, data_set: pd.DataFrame, data_folder: str, asset_id: str):
        # Le a estrategia do arquivo de confgurcao
        with open(self.strategy_file, "r") as f:
            exporter = ExportToParquet()
            startegies = json.load(f)
            for strategy in startegies.keys():
                df_ret = data_set.copy(deep=True)
                folder = os.path.join(data_folder, strategy)
                print(f"Processing strategy: {startegies[strategy]['description']}")
                for function in startegies[strategy]["functions"]:
                    dict_aux = startegies[strategy]["functions"]
                    print(f"Calculating {function}...")
                    if "params" in dict_aux[function]:
                        df_ret = self.apply_function(df_ret, dict_aux[function]['function'], **dict_aux[function]['params'])
                    else:
                        df_ret = self.apply_function(df_ret, dict_aux[function]['function'])

                exporter.export(df_ret, folder_path=folder, file_name=asset_id)
