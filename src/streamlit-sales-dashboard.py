import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

import calendar

# ---- Configuração da página ----
st.set_page_config(
    page_title="Dashboard de Inteligência Comercial",
    page_icon="📊",
    layout="wide",
)

# ---- CSS customizado ----
st.markdown("""
<style>
  .main { padding-top: 2rem; }
  div[data-testid="metric-container"] {
      background-color: white;
      padding: 1rem;
      border-radius: 0.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
  }
</style>
""", unsafe_allow_html=True)

# ---- Função de formatação de mês ----
def format_month(s):
    meses = {
      '01':'Jan','02':'Fev','03':'Mar','04':'Abr',
      '05':'Mai','06':'Jun','07':'Jul','08':'Ago',
      '09':'Set','10':'Out','11':'Nov','12':'Dez'
    }
    if isinstance(s, str) and len(s) >= 7:
        return f"{meses[s[5:7]]}/{s[2:4]}"
    return s

# ---- Carrega as bases processadas ----
BASE_DIR     = Path(__file__).resolve().parent.parent
PROCESSED_DIR= BASE_DIR / "data" / "processed"

sell_in  = pd.read_csv(
    PROCESSED_DIR / "sell_in_processed.csv",
    sep=";", decimal=",", encoding="utf-8-sig"
)
sell_out = pd.read_csv(
    PROCESSED_DIR / "sell_out_processed.csv",
    sep=";", decimal=",", encoding="utf-8-sig"
)

# ---- Título ----
st.markdown("""
<div style="background: linear-gradient(135deg, #ec4899, #8b5cf6);
            padding:2rem; border-radius:0.5rem; margin-bottom:2rem;">
  <h1 style="color:white; margin:0;">Dashboard de Inteligência Comercial</h1>
  <p style="color:white; opacity:0.9; margin:0;">FMCG Doces e Guloseimas</p>
</div>
""", unsafe_allow_html=True)

# ---- Cria as três tabs ----
tab1, tab2, tab3 = st.tabs([
  "📊 Análise de Vendas & KPIs",
  "🔮 Previsão de Vendas",
  "💡 Recomendações Estratégicas"
])

# ==== TAB 1: EDA & KPIs ====
with tab1:
    # Ajusta datas e meses
    sell_in['Data_SellIn'] = pd.to_datetime(
        sell_in['Data_SellIn'], errors='coerce'
    )
    sell_in['Month'] = sell_in['Data_SellIn'].dt.to_period('M').astype(str)

    sell_out['Data'] = pd.to_datetime(
        sell_out['Data'], errors='coerce'
    )
    sell_out['Month'] = sell_out['Data'].dt.to_period('M').astype(str)

    # KPIs agregados
    total_in   = sell_in['Valor_Total'].sum()
    volume_in  = sell_in['Quantidade'].sum()
    ticket_in  = (total_in/volume_in) if volume_in else 0

    total_out  = sell_out['Valor_SellThrough'].sum()
    volume_out = sell_out['Unidades_SellThrough'].sum()
    ticket_out = (total_out/volume_out) if volume_out else 0

    rate        = (total_out/total_in*100) if total_in else 0

    def pct_growth(df, col):
        m = df.groupby('Month')[col].sum().sort_index()
        if len(m) >= 6:
            return ((m[-3:].sum() - m[-6:-3].sum())
                    / m[-6:-3].sum() * 100)
        return 0

    growth_in  = pct_growth(sell_in, 'Valor_Total')
    growth_out = pct_growth(sell_out, 'Valor_SellThrough')

    regions = sell_in['Regiao'].nunique()

    # Métricas Sell-In
    st.markdown("### Sell In")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total (Mi R$)", f"{total_in/1e6:.1f} Mi")
    c2.metric("Volume (Cx)",    f"{volume_in:,.0f}".replace(",","."))  
    c3.metric("Ticket Médio",    f"R$ {ticket_in:,.2f}"
                                  .replace(",", "X")
                                  .replace(".", ",")
                                  .replace("X", "."))
    cresc_in = f"{growth_in:,.1f}".replace(".", ",") + "%"
    c4.metric("Crescimento", cresc_in, delta=cresc_in,
              delta_color="normal" if growth_in>0 else "inverse")

    # Métricas Sell-Out
    st.markdown("### Sell Out")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total (Mi R$)", f"{total_out/1e6:.1f} Mi")
    c2.metric("Volume (Unid)", f"{volume_out:,.0f}".replace(",","."))  
    c3.metric("Ticket Médio",   f"R$ {ticket_out:,.2f}"
                                  .replace(",", "X")
                                  .replace(".", ",")
                                  .replace("X", "."))
    cresc_out = f"{growth_out:,.1f}".replace(".", ",") + "%"
    c4.metric("Crescimento", cresc_out, delta=cresc_out,
              delta_color="normal" if growth_out>0 else "inverse")

    # KPI Geral Sell-Through
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
                color:white;padding:1rem;border-radius:0.5rem;margin:2rem 0;">
      <div style="display:flex;justify-content:space-between">
        <div>
          <h3>🎯 Sell-Through Rate</h3>
          <p style="font-size:2rem;margin:0;">{rate:.1f}%</p>
          <small>Meta: 85%</small>
        </div>
        <div style="text-align:right">
          <small>Regiões</small>
          <p style="font-size:1.5rem;margin:0;">{regions}</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Gráficos de série temporal & comparação
    mi = sell_in.groupby('Month')['Valor_Total'].sum().reset_index()
    mo = sell_out.groupby('Month')['Valor_SellThrough'].sum().reset_index()
    dfm = mi.merge(mo, on='Month', how='outer').fillna(0)
    dfm['Rate'] = (dfm['Valor_SellThrough']
                   /dfm['Valor_Total']*100).clip(0,150)
    dfm = dfm[dfm['Month'] >= '2023-01']
    dfm['M'] = dfm['Month'].apply(format_month)

    a1,a2 = st.columns(2)
    with a1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dfm['M'], y=dfm['Rate'],
            mode='lines+markers+text',
            text=[f"{v:.0f}%" for v in dfm['Rate']],
            textposition="top center",
            line=dict(color='#10b981',width=3,shape='spline'),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=dfm['M'], y=[85]*len(dfm),
            mode='lines',
            line=dict(color='#ef4444',dash='dash'),
            name='Meta 85%'
        ))
        fig.update_layout(
            title="Sell-Through Rate Mês a Mês",
            yaxis=dict(range=[0,150],ticksuffix='%'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with a2:
        dfm['SellInMi']  = dfm['Valor_Total']/1e6
        dfm['SellOutMi'] = dfm['Valor_SellThrough']/1e6
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dfm['M'], y=dfm['SellInMi'],
            mode='lines+markers+text',
            text=[f"{v:.1f} Mi" for v in dfm['SellInMi']],
            textposition='top center',
            line=dict(color='#8884d8',width=3,shape='spline'),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=dfm['M'], y=dfm['SellOutMi'],
            mode='lines+markers+text',
            text=[f"{v:.1f} Mi" for v in dfm['SellOutMi']],
            textposition='bottom center',
            line=dict(color='#82ca9d',width=3,shape='spline'),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Sell In vs Sell Out (Milhões de R$)",
            xaxis=dict(tickangle=45,title="Mês/Ano"),
            yaxis=dict(title="Milhões de R$",
                       tickformat=".1f",ticksuffix=" Mi"),
            hovermode='x unified',
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Roscas
    st.markdown("---")
    r1,r2 = st.columns(2)
    with r1:
        top5 = (sell_in
                .groupby('Tipo_Produto')['Valor_Total']
                .sum()
                .nlargest(5)
                .reset_index())
        pie = px.pie(
            top5,
            names='Tipo_Produto',
            values='Valor_Total',
            hole=0.6,
            title="Share Tipo Produto",
            template="plotly_white"
        )
        pie.update_traces(textinfo="percent")
        st.plotly_chart(pie, use_container_width=True)

    with r2:
        byreg = sell_in.groupby('Regiao')['Valor_Total'].sum().reset_index()
        pie = px.pie(
            byreg,
            names='Regiao',
            values='Valor_Total',
            hole=0.6,
            title="Distribuição por Região",
            template="plotly_white"
        )
        pie.update_traces(textinfo="percent")
        st.plotly_chart(pie, use_container_width=True)


# ==== TAB 2: Forecast & Métricas de Confiabilidade ====
with tab2:


    st.markdown("### 🔮 Forecast Estatístico")

    # --- Dados históricos em Mi R$
    mi = sell_in.groupby('Month')['Valor_Total'].sum() / 1e6
    mo = sell_out.groupby('Month')['Valor_SellThrough'].sum() / 1e6
    df = pd.DataFrame({
        'Month':     mi.index,
        'SellOutMi': mo.reindex(mi.index, fill_value=0).values,
        'SellInMi':  mi.values
    }).reset_index(drop=True)
    df['ds'] = pd.to_datetime(df['Month'] + '-01')
    df['M']  = df['Month'].apply(format_month)

    # --- Próximos 6 meses
    last_ds      = df['ds'].iat[-1]
    future_dates = [last_ds + pd.DateOffset(months=i) for i in range(1,7)]
    future_M     = [format_month(d.strftime('%Y-%m')) for d in future_dates]

    # --- Configurações
    cor_out = "rgba(136,136,216,0.6)"
    cor_in  = "rgba(16,185,129,0.6)"
    metodo  = st.selectbox("Método estatístico:", ["Média Móvel (SMA)", "Média Móvel Exponencial (EWMA)"])

    # --- Calcula SMA ou EWMA + extensão de 6 meses ---
    hist_out = df['SellOutMi'].tolist()
    hist_in  = df['SellInMi'].tolist()

    if metodo == "Média Móvel (SMA)":
        window = st.slider("Janela (meses)", 2, 12, 3, key="sma_win")
        # histórico de SMA
        sma_out = pd.Series(hist_out).rolling(window).mean().tolist()
        sma_in  = pd.Series(hist_in).rolling(window).mean().tolist()
        # estende iterativamente para 6 meses
        for _ in range(6):
            nxt_out = np.mean(hist_out[-window:])
            nxt_in  = np.mean(hist_in[-window:])
            sma_out.append(nxt_out)
            sma_in.append(nxt_in)
            hist_out.append(nxt_out)
            hist_in.append(nxt_in)

        line_out = sma_out
        line_in  = sma_in
        title    = f"SMA ({window} meses) estendida 6m"

    else:
        span = st.slider("Span EWMA", 2, 12, 3, key="ewma_sp")
        # histórico de EWMA
        ewma_out = pd.Series(hist_out).ewm(span=span, adjust=False).mean().tolist()
        ewma_in  = pd.Series(hist_in).ewm(span=span, adjust=False).mean().tolist()
        # estende com último valor para 6 meses
        last_out = ewma_out[-1]
        last_in  = ewma_in[-1]
        ewma_out += [last_out]*6
        ewma_in  += [last_in]*6

        line_out = ewma_out
        line_in  = ewma_in
        title    = f"EWMA (span={span}) estendida 6m"

    # --- Gráfico 1: histórico + forecast da média ---
    M_all = df['M'].tolist() + future_M
    fig1 = go.Figure()
    # Sell Out real
    fig1.add_trace(go.Scatter(
        x=df['M'], y=df['SellOutMi'],
        mode='lines+markers', name='Sell Out (Real)',
        line=dict(color='#8884d8', shape='spline', width=3),
        marker=dict(size=6)
    ))
    # Sell In real
    fig1.add_trace(go.Scatter(
        x=df['M'], y=df['SellInMi'],
        mode='lines+markers', name='Sell In (Real)',
        line=dict(color='#10b981', shape='spline', width=3),
        marker=dict(size=6)
    ))
    # Média móvel estendida Out
    fig1.add_trace(go.Scatter(
        x=M_all, y=line_out,
        mode='lines', name=f"{metodo} Out",
        line=dict(color=cor_out, shape='spline', dash='dash', width=3)
    ))
    # Média móvel estendida In
    fig1.add_trace(go.Scatter(
        x=M_all, y=line_in,
        mode='lines', name=f"{metodo} In",
        line=dict(color=cor_in, shape='spline', dash='dash', width=3)
    ))
    fig1.update_layout(
        title=title,
        xaxis_title="Mês/Ano",
        yaxis_title="Milhões de R$",
        hovermode='x unified',
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Gráfico 2: simulação de Sell-Through desejada ---
    st.markdown("### 🔧 Simulação de Sell-Through Desejada (Jan/25–Jun/25)")

    left, right = st.columns(2)
    desired = []
    # 3 sliders na coluna esquerda (Jan/25–Mar/25)
    for i in range(3):
        with left:
            rate = st.slider(f"{future_M[i]}", 0, 200, 85, key=f"st_l_{i}")
            desired.append(rate/100)
    # 3 sliders na coluna direita (Apr/25–Jun/25)
    for i in range(3, 6):
        with right:
            rate = st.slider(f"{future_M[i]}", 0, 200, 85, key=f"st_r_{i}")
            desired.append(rate/100)

    # usa os 6 valores finais de linha In como forecast de Sell In
    forecast_in = line_in[-6:]
    sim_out     = [forecast_in[i] * desired[i] for i in range(6)]

    fig2 = go.Figure()
    # Sell In (forecast)
    fig2.add_trace(go.Scatter(
        x=future_M, y=forecast_in,
        mode='lines+markers+text', name='Sell In (Forecast)',
        line=dict(color='#10b981', shape='spline', width=3),
        marker=dict(size=6),
        text=[f"{v:.1f} Mi" for v in forecast_in],
        textposition='top center'
    ))
    # Sell Out simulado
    fig2.add_trace(go.Scatter(
        x=future_M, y=sim_out,
        mode='lines+markers+text', name='Sell Out (Simulado)',
        line=dict(color='#8884d8', shape='spline', dash='dash', width=3),
        marker=dict(size=6),
        text=[f"{v:.1f} Mi" for v in sim_out],
        textposition='bottom center'
    ))
    fig2.update_layout(
        title="Simulação: Sell In x Sell Out para atingir Sell-Through",
        xaxis_title="Mês/Ano",
        yaxis_title="Milhões de R$",
        hovermode='x unified',
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)


import os
from PIL import Image

# ==== TAB 3: Recomendações ====
with tab3:
    # caminho relativo ao seu script
    img_path = os.path.join(os.path.dirname(__file__), "recomends.png")  # ajuste o nome do arquivo

    if os.path.exists(img_path):
        img = Image.open(img_path)
        # exibe em full width, mantendo boa resolução para leitura
        st.image(
            img,
            caption="Recomendações Estratégicas",
            use_column_width=True
        )
    else:
        st.error(f"Arquivo de recomendações não encontrado em {img_path}")


# ---- Footer ----
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>Desenvolvido com Streamlit + Plotly</p>",
    unsafe_allow_html=True
)
