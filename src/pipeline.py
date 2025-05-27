import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

class TimeSeriesFeatures(BaseEstimator, TransformerMixin):
    """
    Cria features: time_idx, sazonalidade mensal, lag e rolling mean.
    Espera um DataFrame com colunas ['ds','SellOutMi'].
    """
    def __init__(self, lags=[1,3], rolling_window=3):
        self.lags = lags
        self.rolling = rolling_window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['time_idx']  = np.arange(len(df))
        df['month']     = df['ds'].dt.month
        df['month_sin'] = np.sin(2*np.pi*df['month']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month']/12)

        for lag in self.lags:
            df[f'lag_{lag}'] = df['SellOutMi'].shift(lag).fillna(method='bfill')
        df[f'roll_{self.rolling}'] = (
            df['SellOutMi']
              .rolling(self.rolling, min_periods=1)
              .mean()
        )

        return df[[
            'time_idx','month_sin','month_cos',
            *[f'lag_{lag}' for lag in self.lags],
            f'roll_{self.rolling}'
        ]].values

def build_pipeline(random_state=42):
    return Pipeline([
        ('features', TimeSeriesFeatures(lags=[1,3], rolling_window=3)),
        ('model', RandomForestRegressor(n_estimators=200,
                                        max_depth=5,
                                        random_state=random_state))
    ])

def backtest_and_forecast(df_hist, n_test=6, n_forecast=6):
    """
    df_hist: DataFrame c/ col ds (datetime) e SellOutMi
    Retorna: métricas dict, df_fut (M, SellOutMi, SellInMi), df_all
    """
    # split
    tss = TimeSeriesSplit(n_splits=3)
    X = df_hist[['ds','SellOutMi']].copy()
    y = df_hist['SellOutMi'].values
    # train/test
    split = len(df_hist) - n_test
    train, test = df_hist.iloc[:split], df_hist.iloc[split:]
    # pipeline
    pipe = build_pipeline()
    # backtest CV p/ avaliar (opcional)
    # for train_idx, val_idx in tss.split(train):
    #     pipe.fit(train.iloc[train_idx], train['SellOutMi'].iloc[train_idx])
    #     ...

    # fit final
    pipe.fit(train, train['SellOutMi'])
    y_pred = pipe.predict(test)
    mape    = mean_absolute_percentage_error(test['SellOutMi'], y_pred)
    r2      = r2_score(test['SellOutMi'], y_pred)
    margin  = mape * 100
    accuracy= 100 - margin

    # forecast
    last = df_hist['ds'].iloc[-1]
    future_dates = [last + pd.DateOffset(months=i) for i in range(1, n_forecast+1)]
    df_fut = pd.DataFrame({
        'ds': future_dates,
        'SellOutMi': pipe.predict(pd.DataFrame({
            'ds': future_dates,
            'SellOutMi': [df_hist['SellOutMi'].iloc[-1]]*n_forecast
        }))
    }).assign(SellOutMi=lambda d: d['SellOutMi'].clip(lower=0))

    # ajusta SellIn pela média histórica de SellOut/SellIn
    avg_rate = (df_hist['SellOutMi'] / df_hist['SellInMi']).mean()
    df_fut['SellInMi'] = df_fut['SellOutMi'] / avg_rate

    # monta df_all
    df_hist_plot = df_hist.rename(columns={'SellOutMi':'SellOutMi','SellInMi':'SellInMi'})
    df_fut_plot  = df_fut.rename(columns={'ds':'ds','SellOutMi':'SellOutMi','SellInMi':'SellInMi'})
    df_hist_plot['M'] = df_hist_plot['ds'].dt.strftime('%b/%y')
    df_fut_plot['M']  = df_fut_plot['ds'].dt.strftime('%b/%y')
    df_all = pd.concat([df_hist_plot[['M','SellOutMi','SellInMi']],
                        df_fut_plot [['M','SellOutMi','SellInMi']]],
                       ignore_index=True)

    return {
      'accuracy':accuracy,'margin':margin,
      'r2':r2,'mape':margin
    }, df_fut_plot, df_all
