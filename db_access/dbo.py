from operator import ge
from db_access.sql_util import get_session
from sqlalchemy.sql import text
import pandas as pd
from abc import ABC, abstractmethod

class AbsDbReader(ABC):
    def __init__(self, str_conn) -> None:
        super().__init__()
        self.str_conn = str_conn

    @abstractmethod
    def select(self):
        pass


class StockHistory(AbsDbReader):
    def __init__(self, str_conn) -> None:
        super().__init__(str_conn)

    def select(self):
        session = get_session(self.str_conn)
        sql_cmd = text("""
                        with tmp_selected_assets as (
                                select ta.id, ta.ticker, avg(real_volume) ,count(*)
                                from tb_stock_asset ta
                                join tb_stock_price tp on tp.stock_asset_id = ta.id
                                where tp.dt_price > (select max(dt_price) from tb_stock_price) - interval '1 year'
                                group by 1
                                having count(*) = 250
                                and avg(real_volume) > 200000
                            )
                            select tsa.ticker, tp.dt_price, tp.vl_open as open, tp.vl_close as close, 
                                tp.vl_high as high, tp.vl_low as low, tp.real_volume as volume,
                                COALESCE(ti.market, 'N/D') as market, ti.segment_id, ts.subsector_id, tss.economical_sector_id
                            from tmp_selected_assets tsa
                            join tb_stock_price tp on tp.stock_asset_id = tsa.id
                            join tb_stock_asset sa on tsa.id = sa.id
                            join tb_issuer ti on ti.id = sa.issuer_id
                            join tb_segment ts on ts.id = ti.segment_id
                            join tb_subsector tss on tss.id = ts.subsector_id
                            where tp.dt_price > (select max(dt_price) from tb_stock_price) - interval '10 years'
                            order by tsa.ticker, tp.dt_price
                  """)
        
        return pd.read_sql(sql_cmd, session.bind)

