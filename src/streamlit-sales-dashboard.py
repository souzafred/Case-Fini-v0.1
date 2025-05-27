import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

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
    # (todo o seu bloco de forecast permanece idêntico ao que já estava,
    # sem o menu lateral de upload)

    if sell_in is None or sell_out is None:
        st.warning("⚠️ Bases não encontradas.")
    else:
        st.markdown("### 🔮 Forecast & Métricas de Confiabilidade (Modelos ML)")

        # 1) Escolha do algoritmo
        alg = st.selectbox(
            "Selecione o algoritmo:",
            ["Linear Regression", "Random Forest", "Gradient Boosting"]
        )

        # 2) Série histórica em milhões
        mi = sell_in.groupby('Month')['Valor_Total'].sum() / 1e6
        mo = sell_out.groupby('Month')['Valor_SellThrough'].sum() / 1e6
        df = pd.DataFrame({
            'Month':     mi.index,
            'SellInMi':  mi.values,
            'SellOutMi': mo.reindex(mi.index, fill_value=0).values
        }).reset_index(drop=True)
        df['ds']      = pd.to_datetime(df['Month'] + '-01')
        df['M']       = df['Month'].apply(format_month)

        # 3) Calcular taxa média de sell-through
        df['Rate']    = df['SellOutMi'] / df['SellInMi']
        avg_rate      = df['Rate'].mean()

        # 4) Criar features temporais
        df['time_idx']  = np.arange(len(df))
        df['month']     = df['ds'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 5) Train/test split (últimos 6 meses para validação)
        feats = ['time_idx','month_sin','month_cos']
        X = df[feats].values
        y = df['SellOutMi'].values

        if len(df) > 6:
            split = len(df) - 6
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
        else:
            X_train, y_train = X, y
            X_test, y_test   = np.empty((0,3)), np.empty((0,))

        # 6) Inicializa e treina o modelo
        if alg == "Linear Regression":
            model = LinearRegression()
        elif alg == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingRegressor(random_state=42)

        model.fit(X_train, y_train)

        # 7) Avalia no conjunto de teste
        if len(X_test):
            y_pred_test = model.predict(X_test)
            mape    = mean_absolute_percentage_error(y_test, y_pred_test)
            r2      = r2_score(y_test, y_pred_test)
            margin  = mape * 100
            accuracy= 100 - margin
        else:
            mape = r2 = margin = accuracy = np.nan

        # 8) Forecast para os próximos 6 meses
        last_date    = df['ds'].iloc[-1]
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1,7)]
        fut = pd.DataFrame({'ds': future_dates})
        fut['time_idx']  = np.arange(len(df), len(df) + 6)
        fut['month']     = fut['ds'].dt.month
        fut['month_sin'] = np.sin(2 * np.pi * fut['month'] / 12)
        fut['month_cos'] = np.cos(2 * np.pi * fut['month'] / 12)

        Xf       = fut[feats].values
        out_pred = model.predict(Xf).clip(min=0)
        in_pred  = out_pred / avg_rate
        future_M = [format_month(d.strftime('%Y-%m')) for d in future_dates]

        df_hist = df[['M','SellOutMi','SellInMi']].copy()
        df_fut  = pd.DataFrame({
            'M':         future_M,
            'SellOutMi': out_pred,
            'SellInMi':  in_pred
        })
        df_all  = pd.concat([df_hist, df_fut], ignore_index=True)

        # 9) Exibe métricas de confiabilidade
        st.markdown("#### 📈 Métricas de Confiabilidade")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precisão do Modelo", f"{accuracy:.0f}%" if not np.isnan(accuracy) else "–")
        c2.metric("Margem de Erro",    f"±{margin:.0f}%")
        c3.metric("R² Score",          f"{r2:.2f}" if not np.isnan(r2) else "–")
        c4.metric("MAPE",              f"{margin:.1f}%" if not np.isnan(margin) else "–")

        # 10) Explicação
        with st.expander("❔ O que significam estas métricas?"):
            st.markdown("""
            **📊 Precisão do Modelo**  
            - 100% – Margem de Erro; quanto mais próximo de 100%, melhor.  
            - *Referência:* ≥ 85% é bom; ≥ 90% é excelente.

            **⚠️ Margem de Erro (MAPE)**  
            - Erro médio percentual absoluto entre previsão e real.  
            - *Referência:* até ±10% (excelente), ±10–20% (aceitável), > 30% (precisa melhorar).

            **📈 R² Score**  
            - Proporção da variância explicada pelo modelo.  
            - *Referência:* ≥ 0,8 (ótimo), 0,5–0,8 (razoável), < 0,5 (fraco).

            **🔢 MAPE**  
            - Mesmo conceito de Margem de Erro.  
            - *Referência:* < 10% (excelente), 10–20% (bom), > 20% (precisa ajustes).
            """)

        # 11) Gráfico combinado
        hist = df_all.iloc[:-6]
        fut  = df_all.iloc[-6:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist['M'], y=hist['SellOutMi'],
            mode='lines+markers', name='Sell Out (Histórico)',
            line=dict(color='#82ca9d',width=3,shape='spline'),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=fut['M'], y=fut['SellOutMi'],
            mode='lines+markers+text', name='Sell Out (Forecast)',
            line=dict(color='#82ca9d',width=3,shape='spline',dash='dash'),
            text=[f"{v:.1f} Mi" for v in fut['SellOutMi']],
            textposition='top center'
        ))
        fig.add_trace(go.Scatter(
            x=hist['M'], y=hist['SellInMi'],
            mode='lines+markers', name='Sell In (Histórico)',
            line=dict(color='#8884d8',width=3,shape='spline'),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=fut['M'], y=fut['SellInMi'],
            mode='lines+markers+text', name='Sell In (Forecast)',
            line=dict(color='#8884d8',width=3,shape='spline',dash='dash'),
            text=[f"{v:.1f} Mi" for v in fut['SellInMi']],
            textposition='bottom center'
        ))
        fig.update_layout(
            title="Forecast: Sell Out vs Sell In Ajustado",
            xaxis=dict(tickangle=45), yaxis=dict(ticksuffix=" Mi"),
            hovermode='x unified', template="plotly_white", height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # 12) Cards de projeção
        st.markdown("#### 📊 Projeções para os próximos 6 meses")
        for _, row in df_fut.iterrows():
            col1, col2 = st.columns(2)
            col1.metric(f"{row['M']} – Sell Out",         f"{row['SellOutMi']:.1f} Mi")
            col2.metric(f"{row['M']} – Sell In Ajustado", f"{row['SellInMi']:.1f} Mi")

# ==== TAB 3: Recomendações ====
with tab3:
    st.info("💡 Em breve as Recomendações Estratégicas.")

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>Desenvolvido com Streamlit + Plotly</p>",
    unsafe_allow_html=True
)
