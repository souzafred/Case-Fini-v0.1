import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from io import StringIO

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Inteligência Comercial",
    page_icon="📊",
    layout="wide"
)

# CSS customizado
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

# Carregamento de CSVs com cache
@st.cache_data
def load_data(uploaded_file):
    try:
        content = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(content, sep=";", decimal=",", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# Formatação de mês
def format_month(s):
    m = {'01':'Jan','02':'Fev','03':'Mar','04':'Abr','05':'Mai','06':'Jun',
         '07':'Jul','08':'Ago','09':'Set','10':'Out','11':'Nov','12':'Dez'}
    if isinstance(s,str) and len(s)>=7:
        return f"{m[s[5:7]]}/{s[2:4]}"
    return s

# Título
st.markdown("""
<div style="background: linear-gradient(135deg, #ec4899, #8b5cf6);
            padding:2rem; border-radius:0.5rem; margin-bottom:2rem;">
  <h1 style="color:white;">Dashboard de Inteligência Comercial</h1>
  <p style="color:white;opacity:0.9;">FMCG Doces e Guloseimas</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Upload de Dados")
    in_file  = st.file_uploader("Sell In",  type="csv")
    out_file = st.file_uploader("Sell Out", type="csv")
    if in_file and out_file:
        st.success("Arquivos carregados!")

sell_in  = load_data(in_file)   if in_file  else None
sell_out = load_data(out_file)  if out_file else None

tab1, tab2, tab3 = st.tabs([
    "📊 Análise de Vendas & KPIs",
    "🔮 Previsão de Vendas",
    "💡 Recomendações Estratégicas"
])

with tab1:
    if sell_in is None or sell_out is None:
        st.warning("⚠️ Faça upload de ambos os arquivos para ver o dashboard.")
    else:
        # --- Ajuste de datas e mês
        sell_in['Data_SellIn'] = pd.to_datetime(sell_in['Data_SellIn'], errors='coerce')
        sell_in['Month']       = sell_in['Data_SellIn'].dt.to_period('M').astype(str)
        sell_out['Data']       = pd.to_datetime(sell_out['Data'], errors='coerce')
        sell_out['Month']      = sell_out['Data'].dt.to_period('M').astype(str)

        # --- KPIs agregados
        total_in   = sell_in['Valor_Total'].sum()
        volume_in  = sell_in['Quantidade'].sum()
        ticket_in  = total_in/volume_in if volume_in else 0

        total_out  = sell_out['Valor_SellThrough'].sum()
        volume_out = sell_out['Unidades_SellThrough'].sum()
        ticket_out = total_out/volume_out if volume_out else 0

        rate = (total_out/total_in*100) if total_in else 0

        def pct_growth(df, col):
            m = df.groupby('Month')[col].sum().sort_index()
            if len(m)>=6:
                return ((m[-3:].sum()-m[-6:-3].sum())/m[-6:-3].sum()*100)
            return 0
        growth_in  = pct_growth(sell_in,   'Valor_Total')
        growth_out = pct_growth(sell_out, 'Valor_SellThrough')

        regions = sell_in['Regiao'].nunique()

        # --- Métricas (TOTAL EM MILHÕES)
        st.markdown("### Sell In")
        c1, c2, c3, c4 = st.columns(4)

        # Total em milhões, 1 casa decimal
        c1.metric(
            "Total (Mi R$)",
            f"{total_in/1e6:.1f} Mi"
        )
        c2.metric(
            "Volume (Cx)",
            f"{volume_in:,.0f}".replace(",", ".")
        )
        c3.metric(
            "Ticket Médio",
            f"R$ {ticket_in:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        cres_in_str = f"{growth_in:,.1f}".replace(".", ",") + "%"
        c4.metric(
            "Crescimento",
            cres_in_str,
            delta=cres_in_str,
            delta_color="normal" if growth_in > 0 else "inverse"
        )

        st.markdown("### Sell Out")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total (Mi R$)",
            f"{total_out/1e6:.1f} Mi"
        )
        c2.metric(
            "Volume (Unid)",
            f"{volume_out:,.0f}".replace(",", ".")
        )
        c3.metric(
            "Ticket Médio",
            f"R$ {ticket_out:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        cres_out_str = f"{growth_out:,.1f}".replace(".", ",") + "%"
        c4.metric(
            "Crescimento",
            cres_out_str,
            delta=cres_out_str,
            delta_color="normal" if growth_out > 0 else "inverse"
        )

        # --- KPI Geral Sell-Through
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

        # --- Gráficos (mantém como antes)
        mi = sell_in.groupby('Month')['Valor_Total'].sum().reset_index()
        mo = sell_out.groupby('Month')['Valor_SellThrough'].sum().reset_index()
        dfm = mi.merge(mo, on='Month', how='outer').fillna(0)
        dfm['Rate'] = (dfm['Valor_SellThrough']/dfm['Valor_Total']*100).clip(0,150)
        dfm = dfm[dfm['Month']>='2023-01']
        dfm['M'] = dfm['Month'].apply(format_month)

        a1, a2 = st.columns(2)
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
            dfm['SellInMi']  = dfm['Valor_Total']       / 1e6
            dfm['SellOutMi'] = dfm['Valor_SellThrough'] / 1e6

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
        r1, r2 = st.columns(2)
        with r1:
            top5 = sell_in.groupby('Tipo_Produto')['Valor_Total'].sum().nlargest(5).reset_index()
            fig = px.pie(top5, names='Tipo_Produto', values='Valor_Total', hole=0.6,
                         title="Share Tipo Produto", template="plotly_white")
            fig.update_traces(textinfo="percent")
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            byreg = sell_in.groupby('Regiao')['Valor_Total'].sum().reset_index()
            fig = px.pie(byreg, names='Regiao', values='Valor_Total', hole=0.6,
                         title="Distribuição por Região", template="plotly_white")
            fig.update_traces(textinfo="percent")
            st.plotly_chart(fig, use_container_width=True)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from prophet import Prophet

with tab2:
    if sell_in is None or sell_out is None:
        st.warning("⚠️ Faça upload de ambos os arquivos para ver o Forecast.")
    else:
        st.markdown("### 🔮 Forecast Elaborado (ML Features & Prophet)")

        alg = st.selectbox(
            "Selecione o modelo:",
            ["Linear Regression", "Random Forest", "Gradient Boosting", "Prophet"]
        )

        # --- Construção do histórico em Mi
        mi = sell_in.groupby('Month')['Valor_Total'].sum() / 1e6
        mo = sell_out.groupby('Month')['Valor_SellThrough'].sum() / 1e6
        df = pd.DataFrame({
            'Month':     mi.index,
            'SellInMi':  mi.values,
            'SellOutMi': mo.reindex(mi.index, fill_value=0).values
        }).reset_index(drop=True)
        df['ds'] = pd.to_datetime(df['Month'] + '-01')
        df['M']  = df['Month'].apply(format_month)

        # taxa média de Sell-Through
        df['Rate'] = df['SellOutMi'] / df['SellInMi']
        avg_rate   = df['Rate'].mean()

        # features de tempo para ML
        df['time_idx'] = np.arange(len(df))
        df['month']    = df['ds'].dt.month
        df['month_sin']= np.sin(2*np.pi*df['month']/12)
        df['month_cos']= np.cos(2*np.pi*df['month']/12)

        # train/test split (últimos 6 meses)
        X_feat = ['time_idx', 'month_sin', 'month_cos']
        X = df[X_feat].values
        y = df['SellOutMi'].values
        if len(df) > 6:
            split = len(df) - 6
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
        else:
            X_train, y_train = X, y
            X_test, y_test   = np.empty((0,3)), np.empty((0,))

        # Prophet branch
        if alg == "Prophet":
            # prepara dados
            train_df = df.iloc[:-6][['ds','SellOutMi']].rename(columns={'SellOutMi':'y'})
            test_df  = df.iloc[-6:][['ds','SellOutMi']].rename(columns={'SellOutMi':'y'})

            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(train_df)

            # *AQUI É A ALTERAÇÃO* para freq 'MS'
            future = m.make_future_dataframe(periods=6, freq='MS')
            fcst   = m.predict(future)

            # métricas de teste
            pred_test = fcst.set_index('ds').loc[test_df['ds'], 'yhat'].values
            mape      = mean_absolute_percentage_error(test_df['y'], pred_test)
            r2        = r2_score(test_df['y'], pred_test)
            margin    = mape * 100
            accuracy  = 100 - margin

            # extrai forecast
            last_fcst = fcst.set_index('ds')['yhat']
            fut_dates = last_fcst.index[-6:]
            out_pred  = last_fcst.loc[fut_dates].values
            in_pred   = out_pred / avg_rate
            future_M  = [format_month(d.strftime('%Y-%m')) for d in fut_dates]

            df_hist = df[['M','SellOutMi','SellInMi']].copy()
            df_fut  = pd.DataFrame({
                'M': future_M,
                'SellOutMi': out_pred,
                'SellInMi':  in_pred
            })
            df_all  = pd.concat([df_hist, df_fut], ignore_index=True)

        else:
            # ML branch (Linear, RF, GB)
            if alg=="Linear Regression":
                model = LinearRegression()
            elif alg=="Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingRegressor(random_state=42)

            model.fit(X_train, y_train)
            if len(X_test):
                y_pred = model.predict(X_test)
                mape   = mean_absolute_percentage_error(y_test, y_pred)
                r2     = r2_score(y_test, y_pred)
                margin = mape*100
                accuracy = 100 - margin
            else:
                mape = r2 = margin = accuracy = np.nan

            # forecast ML para 6 meses
            future_idx = np.arange(len(df), len(df)+6)
            fut_dates  = [df['ds'].iloc[-1] + pd.DateOffset(months=i) for i in range(1,7)]
            fut       = pd.DataFrame({'ds': fut_dates})
            fut['time_idx']  = future_idx
            fut['month']     = fut['ds'].dt.month
            fut['month_sin'] = np.sin(2*np.pi*fut['month']/12)
            fut['month_cos'] = np.cos(2*np.pi*fut['month']/12)

            Xf = fut[X_feat].values
            out_pred = model.predict(Xf).clip(min=0)
            in_pred  = out_pred / avg_rate
            future_M = [format_month(d.strftime('%Y-%m')) for d in fut_dates]

            df_hist = df[['M','SellOutMi','SellInMi']].copy()
            df_fut  = pd.DataFrame({
                'M': future_M,
                'SellOutMi': out_pred,
                'SellInMi':  in_pred
            })
            df_all  = pd.concat([df_hist, df_fut], ignore_index=True)

        # exibe métricas
        st.markdown("#### 📈 Métricas de Confiabilidade")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precisão do Modelo", f"{accuracy:.0f}%" if not np.isnan(accuracy) else "–")
        c2.metric("Margem de Erro",    f"±{margin:.0f}%")
        c3.metric("R² Score",          f"{r2:.2f}")
        c4.metric("MAPE",              f"{margin:.1f}%")

        # gráfico combinado
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_all['M'], y=df_all['SellOutMi'],
            mode='lines+markers+text', name='Sell Out (Mi)',
            line=dict(color='#82ca9d',shape='spline',width=3),
            text=[f"{v:.1f} Mi" for v in df_all['SellOutMi']],
            textposition='top center', marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=df_all['M'], y=df_all['SellInMi'],
            mode='lines+markers+text', name='Sell In Ajustado (Mi)',
            line=dict(color='#8884d8',shape='spline',width=3,dash='dash'),
            text=[f"{v:.1f} Mi" for v in df_all['SellInMi']],
            textposition='bottom center', marker=dict(size=6)
        ))
        fig.update_layout(
            title="Forecast: Sell Out vs Sell In Ajustado",
            xaxis=dict(tickangle=45), yaxis=dict(tickformat=".1f",ticksuffix=" Mi"),
            hovermode='x unified', template="plotly_white", height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # cards de projeção
        st.markdown("#### 📊 Projeções Próximos 6 Meses")
        for _, row in df_fut.iterrows():
            col1, col2 = st.columns(2)
            col1.metric(f"{row['M']} – Sell Out",         f"{row['SellOutMi']:.1f} Mi")
            col2.metric(f"{row['M']} – Sell In Ajustado", f"{row['SellInMi']:.1f} Mi")




with tab3:
    st.info("💡 Em breve as Recomendações Estratégicas.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>Desenvolvido com Streamlit + Plotly</p>",
    unsafe_allow_html=True
)
