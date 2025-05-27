import pandas as pd
import plotly.express as px
from pathlib import Path

def load_data(base_dir: Path):
    data_dir = base_dir / "data" / "processed"
    sell_in = pd.read_csv(
        data_dir / "sell_in_processed.csv",
        sep=";", decimal=",", encoding="utf-8-sig",
        parse_dates=["Data_SellIn"]
    )
    sell_out = pd.read_csv(
        data_dir / "sell_out_processed.csv",
        sep=";", decimal=",", encoding="utf-8-sig",
        parse_dates=["Data"]
    )
    return sell_in, sell_out

def prepare_timeseries(sell_in, sell_out):
    df_sn = sell_in.groupby("Ano_Mes")["Valor_Total"].sum().reset_index(name="SellIn")
    df_so = sell_out.groupby("Ano_Mes")["Valor_SellThrough"].sum().reset_index(name="SellOut")
    df_ts = pd.merge(df_sn, df_so, on="Ano_Mes", how="outer").sort_values("Ano_Mes")
    df_ts["SellThroughRate"] = df_ts["SellOut"] / df_ts["SellIn"]
    return df_ts

def plot_sell_through_rate(df_ts):
    fig = px.line(
        df_ts,
        x="Ano_Mes",
        y="SellThroughRate",
        title="Sell Through Rate – Mês a Mês",
        line_shape="spline",             # curvas suaves
        markers=True,                    # marcadores nos pontos
        labels={"SellThroughRate":"Sell-Through Rate","Ano_Mes":"Mês/Ano"},
        template="plotly_white"          # visual limpo
    )
    # adiciona rótulos de % acima de cada ponto
    fig.update_traces(
        mode="lines+markers+text",
        text=[f"{v:.0%}" for v in df_ts["SellThroughRate"]],
        textposition="top center",
        hovertemplate="%{y:.0%}"
    )
    # meta como linha tracejada
    fig.add_hline(
        y=0.85, line_dash="dash", 
        annotation_text="Meta (85%)", annotation_position="top right"
    )
    fig.update_yaxes(
        tickformat=".0%", 
        range=[0, max(df_ts["SellThroughRate"].max()*1.1, 1)]
    )
    fig.update_xaxes(tickangle=45)
    fig.show()

def plot_sell_in_vs_out(df_ts):
    fig = px.line(
        df_ts,
        x="Ano_Mes",
        y=["SellIn","SellOut"],
        title="Sell In vs Sell Out",
        line_shape="spline",
        markers=True,
        labels={"value":"R$","Ano_Mes":"Mês/Ano","variable":"Métrica"},
        template="plotly_white"
    )
    # adiciona rótulos de valor em cada série
    fig.update_traces(
        mode="lines+markers+text",
        textposition="top center"
    )
    # formata texto de cada trace
    for trace in fig.data:
        # trace.y é uma lista de valores do eixo Y
        trace.text = [f"R$ {y:,.0f}" for y in trace.y]
        trace.hovertemplate = "R$ %{y:,.0f}"
    fig.update_xaxes(tickangle=45)
    fig.show()

def main():
    BASE_DIR = Path(__file__).parent
    sell_in, sell_out = load_data(BASE_DIR)
    df_ts = prepare_timeseries(sell_in, sell_out)

    plot_sell_through_rate(df_ts)
    plot_sell_in_vs_out(df_ts)

if __name__ == "__main__":
    main()