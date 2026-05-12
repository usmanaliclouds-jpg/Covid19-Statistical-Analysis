import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
import statsmodels.api as sm
from dash import Dash, dcc, html, Input, Output, State, dash_table
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
df_raw = pd.read_csv('covid_data.csv')
df_raw['date'] = pd.to_datetime(df_raw['date'], dayfirst=True)

# Real countries only (exclude aggregates like "World", "Asia" etc.)
df = df_raw[df_raw['continent'].notna()].copy()
df = df.sort_values(['location', 'date'])

countries  = sorted(df['location'].unique().tolist())
continents = sorted(df['continent'].dropna().unique().tolist())


# ─────────────────────────────────────────────
# 2. COLOUR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    'bg':      '#f0f4f8',
    'card':    '#ffffff',
    'card2':   '#f8fafc',
    'accent1': '#1d6fa4',   # blue
    'accent2': '#dc2626',   # red
    'accent3': '#d97706',   # amber
    'accent4': '#16a34a',   # green
    'accent5': '#7c3aed',   # purple
    'text':    '#1e293b',
    'muted':   '#64748b',
    'border':  '#e2e8f0',
}

CONT_COLORS = {
    'Asia':          '#1d6fa4',
    'Europe':        '#7c3aed',
    'North America': '#d97706',
    'South America': '#16a34a',
    'Africa':        '#dc2626',
    'Oceania':       '#f97316',
}

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(248,250,252,0.9)',
    font=dict(color=COLORS['text'], size=12),
    xaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border']),
    yaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border']),
    legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor=COLORS['border']),
    margin=dict(l=50, r=20, t=50, b=50),
)

# ─────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────
def card(children, style=None):
    base = {
        'background':   COLORS['card'],
        'borderRadius': '12px',
        'padding':      '20px',
        'border':       f'1px solid {COLORS["border"]}',
        'boxShadow':    '0 2px 12px rgba(0,0,0,0.08)',
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def stat_box(label, value, color=COLORS['accent1']):
    return html.Div([
        html.Div(label, style={
            'color': COLORS['muted'], 'fontSize': '11px',
            'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '6px',
        }),
        html.Div(value, style={
            'color': color, 'fontSize': '22px', 'fontWeight': '700',
        }),
    ], style={
        'background':   COLORS['card2'],
        'borderRadius': '10px',
        'padding':      '16px 18px',
        'borderLeft':   f'3px solid {color}',
        'flex':         '1',
        'minWidth':     '160px',
    })


def apply_layout(fig, **kwargs):
    layout = dict(PLOT_LAYOUT)
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


def section_title(text, color=COLORS['accent1']):
    return html.H4(text, style={'color': color, 'marginTop': 0, 'marginBottom': '14px'})


TABLE_STYLE = dict(
    style_header={
        'backgroundColor': '#e2e8f0', 'color': COLORS['accent1'],
        'fontWeight': 'bold', 'fontSize': '12px',
    },
    style_cell={
        'backgroundColor': '#ffffff', 'color': COLORS['text'],
        'fontSize': '12px', 'border': f'1px solid {COLORS["border"]}', 'padding': '8px',
        'textAlign': 'center',
    },
    style_data_conditional=[
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'},
    ],
)

# ─────────────────────────────────────────────
# 4. PRE-COMPUTE GLOBAL STATS (done once at startup)
# ─────────────────────────────────────────────
_snap_global = df.groupby('location').agg(
    total_cases=('total_cases', 'max'),
    total_deaths=('total_deaths', 'max'),
    total_vaccinations=('total_vaccinations', 'max'),
).reset_index()

_total_cases  = _snap_global['total_cases'].sum()
_total_deaths = _snap_global['total_deaths'].sum()
_total_vacc   = _snap_global['total_vaccinations'].sum()
_cfr          = (_total_deaths / _total_cases * 100) if _total_cases else 0
_n_countries  = df['location'].nunique()

_KPI_CHILDREN = [
    stat_box("Total Cases",        f"{_total_cases/1e6:.1f}M",  COLORS['accent3']),
    stat_box("Total Deaths",       f"{_total_deaths/1e6:.2f}M", COLORS['accent2']),
    stat_box("Vaccinations",       f"{_total_vacc/1e9:.2f}B",   COLORS['accent4']),
    stat_box("Case Fatality Rate", f"{_cfr:.2f}%",              COLORS['accent5']),
    stat_box("Countries Tracked",  str(_n_countries),           COLORS['accent1']),
    stat_box("Data Period",        "2020 – 2022",               COLORS['muted']),
]

# ─────────────────────────────────────────────
# 5. APP LAYOUT
# ─────────────────────────────────────────────
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    # ... your other settings
)

# ADD THIS LINE RIGHT HERE
server = app.server
app.title = "COVID-19 Statistical Analysis Dashboard"

TABS_STYLE = {'borderBottom': f'1px solid {COLORS["border"]}', 'background': '#ffffff'}
TAB_S = {
    'background': '#ffffff', 'color': COLORS['muted'], 'border': 'none',
    'padding': '12px 20px', 'fontSize': '13px',
}
TAB_SEL = {**TAB_S, 'color': COLORS['accent1'],
           'borderBottom': f'2px solid {COLORS["accent1"]}', 'background': '#ffffff'}

app.layout = html.Div(style={
    'background': COLORS['bg'], 'minHeight': '100vh', 'color': COLORS['text'],
}, children=[

    # ── HEADER ──────────────────────────────
    html.Div([
        html.Div([
            html.Span("◈ ", style={'color': COLORS['accent1'], 'fontSize': '28px'}),
            html.Span("COVID-19", style={
                'color': COLORS['accent1'], 'fontSize': '24px', 'fontWeight': '800'}),
            html.Span(" Statistical Analysis Dashboard", style={
                'color': COLORS['text'], 'fontSize': '24px'}),
        ]),
        html.Div(
            "Global Pandemic Data · 2020–2022 · 200+ Countries · "
            "Probability & Statistics Semester Project – Spring 2026",
            style={'color': COLORS['muted'], 'fontSize': '12px', 'marginTop': '4px'},
        ),
    ], style={
        'background':    COLORS['card'],
        'padding':       '20px 32px',
        'borderBottom':  f'2px solid {COLORS["accent1"]}',
        'boxShadow':     '0 2px 8px rgba(0,0,0,0.08)',
    }),

    # ── KPI STRIP (static, computed once) ───
    html.Div(_KPI_CHILDREN, style={
        'display': 'flex', 'gap': '12px', 'padding': '18px 24px', 'flexWrap': 'wrap',
    }),

    # ── TABS ────────────────────────────────
    dcc.Tabs(id='tabs', value='overview', style=TABS_STYLE, children=[
        dcc.Tab(label='📊 Overview',     value='overview',     style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label='📈 Time Series',  value='timeseries',   style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label='📋 Descriptive',  value='descriptive',  style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label='🎲 Probability',  value='probability',  style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label='🔗 Regression',   value='regression',   style=TAB_S, selected_style=TAB_SEL),
        dcc.Tab(label='🌍 Compare',      value='compare',      style=TAB_S, selected_style=TAB_SEL),
    ]),

    html.Div(id='tab-content', style={'padding': '20px 24px'}),
])


# ─────────────────────────────────────────────
# 6. TAB ROUTER
# ─────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'overview':    return layout_overview()
    if tab == 'timeseries':  return layout_timeseries()
    if tab == 'descriptive': return layout_descriptive()
    if tab == 'probability': return layout_probability()
    if tab == 'regression':  return layout_regression()
    if tab == 'compare':     return layout_compare()
    return html.Div("Select a tab")


# ═══════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════
def layout_overview():
    snap = df.groupby('location').agg(
        iso_code=('iso_code', 'first'),
        total_cases=('total_cases', 'max'),
        total_deaths=('total_deaths', 'max'),
        continent=('continent', 'first'),
    ).reset_index()

    # World choropleth
    fig_map = px.choropleth(
        snap, locations='iso_code', color='total_cases',
        hover_name='location', color_continuous_scale='Blues',
        title='🌐 Total Confirmed Cases by Country',
    )
    apply_layout(fig_map, coloraxis_colorbar=dict(title='Cases'))

    # Top-15 bar
    top15 = snap.nlargest(15, 'total_cases')
    fig_bar = px.bar(
        top15, x='total_cases', y='location', orientation='h',
        color='total_cases', color_continuous_scale='Blues',
        title='🏆 Top 15 Countries by Total Cases',
    )
    apply_layout(fig_bar, yaxis_categoryorder='total ascending', showlegend=False)

    # Cases by continent donut
    cont_cases = df[df['continent'].notna()].groupby('continent')['new_cases'].sum().reset_index()
    fig_pie = px.pie(
        cont_cases, values='new_cases', names='continent',
        title='🥧 Share of Cases by Continent',
        color='continent', color_discrete_map=CONT_COLORS, hole=0.45,
    )
    apply_layout(fig_pie)

    # Deaths vs Cases scatter
    snap2 = snap[(snap['total_cases'] > 1000) & snap['continent'].notna()]
    fig_sc = px.scatter(
        snap2, x='total_cases', y='total_deaths',
        color='continent', hover_name='location',
        color_discrete_map=CONT_COLORS,
        title='💀 Total Deaths vs Total Cases (log scale)',
        log_x=True, log_y=True,
        size='total_deaths', size_max=30,
    )
    apply_layout(fig_sc)

    return html.Div([
        card(dcc.Graph(figure=fig_map, config={'displayModeBar': False})),
        html.Div([
            card(dcc.Graph(figure=fig_bar, config={'displayModeBar': False})),
            card(dcc.Graph(figure=fig_pie, config={'displayModeBar': False})),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                  'gap': '16px', 'marginTop': '16px'}),
        card(dcc.Graph(figure=fig_sc, config={'displayModeBar': False}),
             {'marginTop': '16px'}),
    ])


# ═══════════════════════════════════════════════════════════
# TAB 2 – TIME SERIES
# ═══════════════════════════════════════════════════════════
def layout_timeseries():
    defaults = [c for c in ['United States', 'India', 'Brazil', 'Germany', 'Pakistan']
                if c in countries]
    return html.Div([
        card([
            html.Div("Select Countries",
                     style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '8px'}),
            dcc.Dropdown(
                id='ts-countries',
                options=[{'label': c, 'value': c} for c in countries],
                value=defaults, multi=True,
                style={'background': COLORS['card2'], 'color': '#000'},
            ),
            html.Div(style={'height': '12px'}),
            html.Div("Select Metric",
                     style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '8px'}),
            dcc.Dropdown(
                id='ts-metric',
                options=[
                    {'label': 'New Cases (daily)',  'value': 'new_cases'},
                    {'label': 'Total Cases',        'value': 'total_cases'},
                    {'label': 'New Deaths (daily)', 'value': 'new_deaths'},
                    {'label': 'Total Deaths',       'value': 'total_deaths'},
                    {'label': 'New Vaccinations',   'value': 'new_vaccinations'},
                    {'label': 'Stringency Index',   'value': 'stringency_index'},
                ],
                value='new_cases',
                style={'background': COLORS['card2'], 'color': '#000'},
            ),
        ]),
        html.Div(style={'height': '16px'}),
        card(dcc.Graph(id='ts-graph', config={'displayModeBar': True})),
        html.Div(style={'height': '16px'}),
        card([
            section_title("7-Day Rolling Average", COLORS['accent3']),
            html.P(
                "A rolling (moving) average smooths out daily noise by averaging "
                "the last 7 days. This makes trends easier to see.",
                style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '10px'}
            ),
            dcc.Graph(id='ts-rolling', config={'displayModeBar': False}),
        ]),
    ])


@app.callback(
    Output('ts-graph', 'figure'),
    Output('ts-rolling', 'figure'),
    Input('ts-countries', 'value'),
    Input('ts-metric', 'value'),
)
def update_ts(selected_countries, metric):
    if not selected_countries:
        return go.Figure(), go.Figure()
    sub = df[df['location'].isin(selected_countries)]
    clr_list = list(CONT_COLORS.values())

    fig  = go.Figure()
    fig2 = go.Figure()
    for i, loc in enumerate(selected_countries):
        d = sub[sub['location'] == loc].copy()
        color = clr_list[i % len(clr_list)]
        fig.add_trace(go.Scatter(
            x=d['date'], y=d[metric], name=loc, mode='lines',
            line=dict(color=color, width=1.5),
        ))
        # min_periods=1 so the first few days are not blank
        d['rolling'] = d[metric].rolling(7, min_periods=1).mean()
        fig2.add_trace(go.Scatter(
            x=d['date'], y=d['rolling'], name=loc, mode='lines',
            line=dict(color=color, width=2),
        ))

    apply_layout(fig,
                 title=f'📈 {metric.replace("_"," ").title()} Over Time',
                 hovermode='x unified')
    apply_layout(fig2,
                 title=f'📉 7-Day Rolling Mean – {metric.replace("_"," ").title()}',
                 hovermode='x unified')
    return fig, fig2


# ═══════════════════════════════════════════════════════════
# TAB 3 – DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════
def layout_descriptive():
    numeric_cols = [
        'new_cases', 'total_cases', 'new_deaths', 'total_deaths',
        'total_vaccinations', 'stringency_index', 'gdp_per_capita',
        'life_expectancy', 'positive_rate', 'hospital_beds_per_thousand',
    ]
    snap = df.groupby('location')[numeric_cols].max().reset_index(drop=True)

    # ── Descriptive table ─────────────────────────────────
    desc = snap[numeric_cols].describe().T.reset_index()
    desc.columns = ['Variable', 'Count', 'Mean', 'Std Dev', 'Min',
                    '25%', 'Median', '75%', 'Max']
    desc['Variable'] = desc['Variable'].str.replace('_', ' ').str.title()
    for col in desc.columns[1:]:
        desc[col] = desc[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")

    # ── 95% Confidence Intervals ──────────────────────────
    ci_rows = []
    ci_vars = ['new_cases', 'total_deaths', 'positive_rate',
               'life_expectancy', 'gdp_per_capita']
    ci_means, ci_lowers, ci_uppers, ci_labels = [], [], [], []

    for col in ci_vars:
        vals = snap[col].dropna()
        n    = len(vals)
        mean = vals.mean()
        se   = stats.sem(vals)
        ci   = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        label = col.replace('_', ' ').title()
        ci_rows.append({
            'Variable':     label,
            'N':            n,
            'Mean':         f"{mean:,.2f}",
            'Std Error':    f"{se:,.2f}",
            '95% CI Lower': f"{ci[0]:,.2f}",
            '95% CI Upper': f"{ci[1]:,.2f}",
            'Interpretation': 'We are 95% confident the true population mean lies in this range.',
        })
        ci_means.append(mean)
        ci_lowers.append(mean - ci[0])
        ci_uppers.append(ci[1] - mean)
        ci_labels.append(label)

    ci_df = pd.DataFrame(ci_rows)

    # ── CI visualisation (error bar chart) ───────────────
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Bar(
        x=ci_labels,
        y=ci_means,
        error_y=dict(
            type='data',
            symmetric=False,
            array=ci_uppers,
            arrayminus=ci_lowers,
            color=COLORS['accent2'],
            thickness=2,
            width=8,
        ),
        marker_color=COLORS['accent1'],
        opacity=0.75,
        name='Mean ± 95% CI',
    ))
    apply_layout(fig_ci,
                 title='📏 95% Confidence Intervals for Key Variables',
                 yaxis_title='Value',
                 xaxis_title='Variable')

    # ── Box plots ──────────────────────────────────────────
    fig_box = make_subplots(rows=1, cols=4,
                             subplot_titles=['New Cases', 'New Deaths',
                                            'Positive Rate', 'Life Expectancy'])
    box_cols   = ['new_cases', 'new_deaths', 'positive_rate', 'life_expectancy']
    box_colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3'], COLORS['accent4']]
    for i, (col, clr) in enumerate(zip(box_cols, box_colors), 1):
        fig_box.add_trace(
            go.Box(y=snap[col].dropna(), name=col.replace('_', ' ').title(),
                   marker_color=clr, boxmean='sd'),
            row=1, col=i,
        )
    apply_layout(fig_box, title='📦 Box Plots of Key Variables', showlegend=False)

    # ── Histogram (new cases) ──────────────────────────────
    fig_hist = go.Figure()
    clip_val = snap['new_cases'].quantile(0.95)
    fig_hist.add_trace(go.Histogram(
        x=snap['new_cases'].clip(0, clip_val), nbinsx=40,
        name='New Cases', marker_color=COLORS['accent1'], opacity=0.8,
    ))
    apply_layout(fig_hist,
                 title='📊 Distribution of New Cases (clipped at 95th percentile)',
                 xaxis_title='New Cases', yaxis_title='Frequency')

    return html.Div([
        card([
            section_title("Descriptive Statistics Summary"),
            html.P(
                "Shows count, mean, standard deviation, min, quartiles, and max for each variable.",
                style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '10px'}
            ),
            dash_table.DataTable(
                data=desc.to_dict('records'),
                columns=[{'name': c, 'id': c} for c in desc.columns],
                **TABLE_STYLE,
            ),
        ]),
        html.Div(style={'height': '16px'}),
        card([
            section_title("95% Confidence Intervals", COLORS['accent3']),
            html.P(
                "A 95% CI means: if we repeated this study 100 times, 95 of those intervals "
                "would contain the true population mean.",
                style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '10px'}
            ),
            dash_table.DataTable(
                data=ci_df.to_dict('records'),
                columns=[{'name': c, 'id': c} for c in ci_df.columns],
                **TABLE_STYLE,
            ),
        ]),
        html.Div(style={'height': '16px'}),
        card(dcc.Graph(figure=fig_ci, config={'displayModeBar': False})),
        html.Div(style={'height': '16px'}),
        card(dcc.Graph(figure=fig_box, config={'displayModeBar': False})),
        html.Div(style={'height': '16px'}),
        card(dcc.Graph(figure=fig_hist, config={'displayModeBar': False})),
    ])


# ═══════════════════════════════════════════════════════════
# TAB 4 – PROBABILITY / DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════
def layout_probability():
    variables = [
        'new_cases', 'total_deaths', 'positive_rate',
        'life_expectancy', 'gdp_per_capita',
    ]
    # Pre-compute default figures so graphs show immediately on tab load
    # without waiting for the dropdown callback to fire
    default_hist, default_qq, default_stats = update_prob('new_cases')

    return html.Div([
        card([
            section_title("Distribution Fitting & Probability Analysis", COLORS['accent5']),
            html.P(
                "Select a variable to fit probability distributions and compute real probabilities.",
                style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '10px'}
            ),
            html.Div("Select Variable", style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '8px'}),
            dcc.Dropdown(
                id='prob-var',
                options=[{'label': v.replace('_', ' ').title(), 'value': v} for v in variables],
                value='new_cases',
                style={'background': COLORS['card2'], 'color': '#000'},
            ),
        ]),
        html.Div(style={'height': '16px'}),
        # Probability calculator
        card([
            section_title("Probability Calculator", COLORS['accent3']),
            html.P(
                "Enter a threshold value to compute: P(X > threshold) and P(X < threshold) "
                "based on the fitted Normal distribution.",
                style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '10px'}
            ),
            html.Div(style={'display': 'flex', 'gap': '12px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                dcc.Input(
                    id='prob-threshold',
                    type='number',
                    placeholder='Enter threshold value...',
                    debounce=True,
                    style={
                        'background': '#f8fafc', 'color': COLORS['text'],
                        'border': f'1px solid {COLORS["border"]}', 'borderRadius': '6px',
                        'padding': '10px 14px', 'fontSize': '14px', 'width': '260px',
                    }
                ),
                html.Button(
                    'Calculate Probability',
                    id='prob-calc-btn',
                    n_clicks=0,
                    style={
                        'background': COLORS['accent1'], 'color': '#ffffff',
                        'border': 'none', 'borderRadius': '6px',
                        'padding': '10px 20px', 'fontWeight': '700',
                        'cursor': 'pointer', 'fontSize': '13px',
                    }
                ),
            ]),
            html.Div(id='prob-calc-result', style={'marginTop': '14px'}),
        ]),
        html.Div(style={'height': '16px'}),
        card(dcc.Graph(id='prob-hist', figure=default_hist, config={'displayModeBar': False})),
        html.Div(style={'height': '16px'}),
        card(dcc.Graph(id='prob-qq',   figure=default_qq,   config={'displayModeBar': False})),
        html.Div(style={'height': '16px'}),
        card(html.Div(id='prob-stats', children=default_stats)),
    ])


@app.callback(
    Output('prob-hist', 'figure'),
    Output('prob-qq',   'figure'),
    Output('prob-stats', 'children'),
    Input('prob-var', 'value'),
)
def update_prob(var):
    snap = df.groupby('location')[var].max().dropna()
    snap = snap[snap > 0]
    vals = snap.values

    # Fit Normal distribution
    mu, sigma = stats.norm.fit(vals)

    # Fit Log-Normal distribution (good for count/skewed data)
    shape, loc_ln, scale_ln = stats.lognorm.fit(vals, floc=0)

    x_range  = np.linspace(vals.min(), vals.max(), 300)
    pdf_norm = stats.norm.pdf(x_range, mu, sigma)
    pdf_logn = stats.lognorm.pdf(x_range, shape, loc_ln, scale_ln)

    # ── Histogram + both fitted distributions ─────────────
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=vals, nbinsx=35, histnorm='probability density',
        name='Empirical Data', marker_color=COLORS['accent1'], opacity=0.55,
    ))
    fig_hist.add_trace(go.Scatter(
        x=x_range, y=pdf_norm, name=f'Normal (μ={mu:.0f}, σ={sigma:.0f})',
        line=dict(color=COLORS['accent2'], width=2.5),
    ))
    fig_hist.add_trace(go.Scatter(
        x=x_range, y=pdf_logn, name='Log-Normal Fit',
        line=dict(color=COLORS['accent3'], width=2.5, dash='dot'),
    ))
    apply_layout(fig_hist,
                 title=f'🔔 Distribution Fitting – {var.replace("_"," ").title()}')

    # ── Q-Q Plot ───────────────────────────────────────────
    (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist='norm')
    line_y = slope * np.array(osm) + intercept
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data',
                                marker=dict(color=COLORS['accent1'], size=5)))
    fig_qq.add_trace(go.Scatter(x=osm, y=line_y, mode='lines', name='Normal Line',
                                line=dict(color=COLORS['accent2'], width=2)))
    apply_layout(fig_qq,
                 title=f'📐 Q-Q Plot – {var.replace("_"," ").title()} '
                       f'(Points close to line = more normal)',
                 xaxis_title='Theoretical Quantiles (what normal expects)',
                 yaxis_title='Sample Quantiles (your data)')

    # ── Normality test ────────────────────────────────────
    n = len(vals)
    if n <= 5000:
        stat_t, p_val = stats.shapiro(vals[:5000])
        test_name = 'Shapiro-Wilk'
    else:
        stat_t, p_val = stats.kstest(vals, 'norm', args=(mu, sigma))
        test_name = 'Kolmogorov-Smirnov'

    skewness = stats.skew(vals)
    kurt     = stats.kurtosis(vals)
    pct_above_mean = (vals > mu).mean() * 100
    within_1s = (np.abs(vals - mu) < sigma).mean() * 100
    within_2s = (np.abs(vals - mu) < 2 * sigma).mean() * 100

    info = [
        section_title(f"Probability Analysis – {var.replace('_',' ').title()}", COLORS['accent5']),
        html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}, children=[
            stat_box("Mean (μ)",        f"{mu:,.1f}",        COLORS['accent1']),
            stat_box("Std Dev (σ)",     f"{sigma:,.1f}",     COLORS['accent3']),
            stat_box("Skewness",        f"{skewness:.3f}",   COLORS['accent2']),
            stat_box("Excess Kurtosis", f"{kurt:.3f}",       COLORS['accent4']),
            stat_box(f"{test_name} p",  f"{p_val:.4f}",      COLORS['accent5']),
            stat_box("% Above Mean",    f"{pct_above_mean:.1f}%", COLORS['muted']),
        ]),
        html.Div(style={
            'marginTop': '16px', 'padding': '14px',
            'background': COLORS['card2'], 'borderRadius': '8px',
            'fontSize': '13px', 'lineHeight': '2',
        }, children=[
            html.Strong("Normality Test: ", style={'color': COLORS['accent1']}),
            html.Span(
                f"{test_name} statistic = {stat_t:.4f}, p = {p_val:.4f}. "
                + ("❌ Data is NOT normally distributed (p < 0.05). "
                   "Log-Normal is likely a better fit."
                   if p_val < 0.05
                   else "✅ Data appears normally distributed (p ≥ 0.05)."),
                style={'color': COLORS['text']},
            ),
            html.Br(),
            html.Strong("Skewness: ", style={'color': COLORS['accent3']}),
            html.Span(
                f"Skewness = {skewness:.3f} → "
                + ("Right-skewed: most countries have low values, but a few have very high values."
                   if skewness > 0.5 else
                   "Left-skewed: most countries have high values."
                   if skewness < -0.5 else
                   "Approximately symmetric (close to normal shape)."),
                style={'color': COLORS['text']},
            ),
            html.Br(),
            html.Strong("Empirical Rule (68–95–99.7): ", style={'color': COLORS['accent4']}),
            html.Span(
                f"Within 1σ: {within_1s:.1f}%  |  Within 2σ: {within_2s:.1f}%  "
                f"(Normal theory predicts 68% and 95%)",
                style={'color': COLORS['text']},
            ),
            html.Br(),
            html.Strong("P(X > μ): ", style={'color': COLORS['accent2']}),
            html.Span(
                f"{pct_above_mean:.1f}% of countries have {var.replace('_',' ')} above the mean. "
                f"(For a perfect Normal distribution, this would be exactly 50%.)",
                style={'color': COLORS['text']},
            ),
        ]),
    ]
    return fig_hist, fig_qq, info


@app.callback(
    Output('prob-calc-result', 'children'),
    Input('prob-calc-btn', 'n_clicks'),
    State('prob-threshold', 'value'),
    State('prob-var', 'value'),
    prevent_initial_call=True,
)
def compute_probability(n_clicks, threshold, var):
    if threshold is None:
        return html.Span("Please enter a threshold value.", style={'color': COLORS['accent2']})

    snap = df.groupby('location')[var].max().dropna()
    snap = snap[snap > 0]
    vals = snap.values
    mu, sigma = stats.norm.fit(vals)

    p_above = 1 - stats.norm.cdf(threshold, mu, sigma)
    p_below = stats.norm.cdf(threshold, mu, sigma)

    # Empirical (actual data) probability
    emp_above = (vals > threshold).mean()
    n_above   = (vals > threshold).sum()
    n_total   = len(vals)

    var_label = var.replace('_', ' ').title()

    return html.Div([
        html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '12px'}, children=[
            stat_box(f"P(X > {threshold:,.0f})\n[Normal]",  f"{p_above:.4f}  ({p_above*100:.1f}%)", COLORS['accent2']),
            stat_box(f"P(X < {threshold:,.0f})\n[Normal]",  f"{p_below:.4f}  ({p_below*100:.1f}%)", COLORS['accent4']),
            stat_box(f"P(X > {threshold:,.0f})\n[Empirical]", f"{emp_above:.4f}  ({emp_above*100:.1f}%)", COLORS['accent3']),
            stat_box(f"Countries Above Threshold", f"{n_above} of {n_total}", COLORS['accent1']),
        ]),
        html.Div(style={
            'padding': '12px', 'background': COLORS['card2'],
            'borderRadius': '8px', 'fontSize': '13px', 'lineHeight': '1.9',
        }, children=[
            html.Strong("Interpretation: ", style={'color': COLORS['accent1']}),
            html.Span(
                f"Based on Normal distribution fit (μ={mu:,.1f}, σ={sigma:,.1f}): "
                f"the probability that a randomly selected country has {var_label} above "
                f"{threshold:,.0f} is {p_above*100:.1f}%. "
                f"In actual data, {n_above} out of {n_total} countries ({emp_above*100:.1f}%) "
                f"exceed this threshold.",
                style={'color': COLORS['text']},
            ),
        ]),
    ])


# ═══════════════════════════════════════════════════════════
# TAB 5 – REGRESSION
# ═══════════════════════════════════════════════════════════
def layout_regression():
    predictors = [
        'gdp_per_capita', 'life_expectancy', 'stringency_index',
        'aged_65_older', 'hospital_beds_per_thousand', 'diabetes_prevalence',
    ]
    targets = ['total_deaths', 'total_cases', 'positive_rate']

    return html.Div([
        card([
            section_title("Simple Linear Regression"),
            html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}, children=[
                html.Div([
                    html.Div("Target Variable (Y)",
                             style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '6px'}),
                    dcc.Dropdown(
                        id='reg-y',
                        options=[{'label': t.replace('_', ' ').title(), 'value': t} for t in targets],
                        value='total_deaths',
                        style={'background': COLORS['card2'], 'color': '#000', 'width': '250px'},
                    ),
                ]),
                html.Div([
                    html.Div("Predictor Variable (X)",
                             style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '6px'}),
                    dcc.Dropdown(
                        id='reg-x',
                        options=[{'label': p.replace('_', ' ').title(), 'value': p} for p in predictors],
                        value='gdp_per_capita',
                        style={'background': COLORS['card2'], 'color': '#000', 'width': '250px'},
                    ),
                ]),
            ]),
        ]),
        html.Div(style={'height': '16px'}),
        card(dcc.Graph(id='reg-scatter', config={'displayModeBar': False})),
        html.Div(style={'height': '16px'}),
        card(html.Div(id='reg-summary')),
        html.Div(style={'height': '16px'}),

        # ── PREDICTION TOOL ────────────────────────────────
        card([
            section_title("🔮 Make a Prediction", COLORS['accent3']),
            html.P(
                "Enter a value for your chosen X variable and the model will predict Y.",
                style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '10px'}
            ),
            html.Div(style={'display': 'flex', 'gap': '12px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                dcc.Input(
                    id='reg-pred-input',
                    type='number',
                    placeholder='Enter X value...',
                    debounce=True,
                    style={
                        'background': '#f8fafc', 'color': COLORS['text'],
                        'border': f'1px solid {COLORS["border"]}', 'borderRadius': '6px',
                        'padding': '10px 14px', 'fontSize': '14px', 'width': '220px',
                    }
                ),
                html.Button(
                    'Predict',
                    id='reg-pred-btn',
                    n_clicks=0,
                    style={
                        'background': COLORS['accent4'], 'color': '#ffffff',
                        'border': 'none', 'borderRadius': '6px',
                        'padding': '10px 24px', 'fontWeight': '700',
                        'cursor': 'pointer', 'fontSize': '13px',
                    }
                ),
            ]),
            html.Div(id='reg-pred-result', style={'marginTop': '14px'}),
        ]),
        html.Div(style={'height': '16px'}),

        card([
            section_title("Multiple Linear Regression — Key Predictors of Total Deaths", COLORS['accent5']),
            html.P(
                "Multiple regression uses several predictor variables (X₁, X₂, …) at once "
                "to explain one outcome (Y). This gives a more complete picture than simple regression.",
                style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '10px'}
            ),
            html.Div(id='multi-reg'),
        ]),
        html.Div(style={'height': '16px'}),
        card([
            section_title("Residuals Plot (Simple Regression)", COLORS['accent3']),
            html.P(
                "Residuals = actual value − predicted value. If residuals are randomly "
                "scattered around zero, the model is appropriate.",
                style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '10px'}
            ),
            dcc.Graph(id='reg-resid', config={'displayModeBar': False}),
        ]),
    ])


@app.callback(
    Output('reg-scatter', 'figure'),
    Output('reg-summary', 'children'),
    Output('multi-reg',   'children'),
    Output('reg-resid',   'figure'),
    Input('reg-x', 'value'),
    Input('reg-y', 'value'),
)
def update_regression(x_col, y_col):
    needed_cols = ['continent', x_col, y_col,
                   'gdp_per_capita', 'life_expectancy', 'stringency_index',
                   'aged_65_older', 'hospital_beds_per_thousand']
    unique_cols = list(dict.fromkeys(needed_cols))

    snap = df.groupby('location')[unique_cols].max().reset_index()
    snap = snap[snap['continent'].notna()].dropna(subset=[x_col, y_col])
    snap = snap[(snap[x_col] > 0) & (snap[y_col] > 0)]

    X_val = snap[x_col].values
    Y_val = snap[y_col].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_val, Y_val)

    x_line = np.linspace(X_val.min(), X_val.max(), 200)
    y_line  = slope * x_line + intercept

    # ── Scatter + regression line ─────────────────────────
    fig_sc = go.Figure()
    for cont in snap['continent'].dropna().unique():
        sub = snap[snap['continent'] == cont]
        fig_sc.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col], mode='markers', name=cont,
            marker=dict(color=CONT_COLORS.get(cont, '#aaa'), size=8, opacity=0.8),
        ))
    fig_sc.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines', name='Regression Line',
        line=dict(color=COLORS['accent2'], width=2.5, dash='dash'),
    ))
    apply_layout(fig_sc,
                 title=f'📍 {x_col.replace("_"," ").title()} vs {y_col.replace("_"," ").title()}',
                 xaxis_title=x_col.replace('_', ' ').title(),
                 yaxis_title=y_col.replace('_', ' ').title())

    # ── Regression summary ────────────────────────────────
    summary = html.Div([
        section_title("Simple Linear Regression Summary"),
        html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}, children=[
            stat_box("Slope (β₁)",      f"{slope:,.4f}",     COLORS['accent1']),
            stat_box("Intercept (β₀)",  f"{intercept:,.2f}", COLORS['accent3']),
            stat_box("R² (Coeff Det.)", f"{r_value**2:.4f}", COLORS['accent4']),
            stat_box("R (Correlation)", f"{r_value:.4f}",    COLORS['accent2']),
            stat_box("p-value",         f"{p_value:.4f}",    COLORS['accent5']),
            stat_box("Std Error",       f"{std_err:,.4f}",   COLORS['muted']),
        ]),
        html.Div(style={
            'marginTop': '14px', 'padding': '12px',
            'background': COLORS['card2'], 'borderRadius': '8px', 'fontSize': '13px',
            'lineHeight': '1.9',
        }, children=[
            html.Strong("Regression Equation: ", style={'color': COLORS['accent1']}),
            html.Span(f"Ŷ = {slope:,.4f} × X + ({intercept:,.2f})",
                      style={'color': COLORS['text']}),
            html.Br(),
            html.Strong("What this means: ", style={'color': COLORS['accent3']}),
            html.Span(
                f"For every 1-unit increase in {x_col.replace('_',' ')}, "
                f"{y_col.replace('_',' ')} changes by {slope:,.4f} on average.",
                style={'color': COLORS['text']},
            ),
            html.Br(),
            html.Strong("Significance: ", style={'color': COLORS['accent2']}),
            html.Span(
                ("✅ Statistically significant (p < 0.05). " if p_value < 0.05
                 else "⚠️ Not statistically significant (p ≥ 0.05). ")
                + f"The model explains {r_value**2*100:.1f}% of the variance in "
                  f"{y_col.replace('_',' ')}.",
                style={'color': COLORS['text']},
            ),
        ]),
    ])

    # ── Multiple regression ───────────────────────────────
    multi_cols = ['gdp_per_capita', 'life_expectancy',
                  'aged_65_older', 'hospital_beds_per_thousand']
    snap_m = df.groupby('location')[['total_deaths'] + multi_cols].max().reset_index(drop=True)
    snap_m = snap_m.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    snap_m = snap_m[snap_m['total_deaths'] > 0]

    X_m   = sm.add_constant(snap_m[multi_cols])
    model = sm.OLS(snap_m['total_deaths'], X_m).fit()

    coef_df = pd.DataFrame({
        'Variable':    ['Intercept'] + [c.replace('_', ' ').title() for c in multi_cols],
        'Coefficient': model.params.values,
        'Std Error':   model.bse.values,
        't-statistic': model.tvalues.values,
        'p-value':     model.pvalues.values,
    })
    coef_df['Significant?'] = coef_df['p-value'].apply(
        lambda p: '✓ Yes (p<0.05)' if p < 0.05 else '✗ No')
    for col in ['Coefficient', 'Std Error', 't-statistic']:
        coef_df[col] = coef_df[col].apply(lambda x: f"{x:,.4f}")
    coef_df['p-value'] = coef_df['p-value'].apply(lambda x: f"{x:.4f}")

    multi_content = html.Div([
        html.Div(style={'display': 'flex', 'gap': '12px', 'marginBottom': '14px', 'flexWrap': 'wrap'},
                 children=[
                     stat_box("R²",           f"{model.rsquared:.4f}",     COLORS['accent1']),
                     stat_box("Adj. R²",      f"{model.rsquared_adj:.4f}", COLORS['accent3']),
                     stat_box("F-statistic",  f"{model.fvalue:.2f}",       COLORS['accent2']),
                     stat_box("Prob(F)",      f"{model.f_pvalue:.4f}",     COLORS['accent4']),
                     stat_box("Observations", f"{int(model.nobs)}",        COLORS['muted']),
                 ]),
        dash_table.DataTable(
            data=coef_df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in coef_df.columns],
            **TABLE_STYLE,
        ),
        html.Div(style={
            'marginTop': '14px', 'padding': '12px',
            'background': COLORS['card2'], 'borderRadius': '8px', 'fontSize': '13px',
            'lineHeight': '1.9',
        }, children=[
            html.Strong("How to read this table: ", style={'color': COLORS['accent1']}),
            html.Span(
                "Each row is one predictor. The Coefficient shows its effect on Total Deaths. "
                "p-value < 0.05 means the predictor is statistically significant. "
                f"Overall model R² = {model.rsquared:.4f}, meaning the model explains "
                f"{model.rsquared*100:.1f}% of variation in total deaths.",
                style={'color': COLORS['text']},
            ),
        ]),
    ])

    # ── Residuals ─────────────────────────────────────────
    y_pred    = slope * X_val + intercept
    residuals = Y_val - y_pred
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode='markers',
        marker=dict(color=COLORS['accent4'], size=7, opacity=0.7),
        name='Residuals',
    ))
    fig_resid.add_hline(y=0, line_color=COLORS['accent2'],
                        line_dash='dash', line_width=2)
    apply_layout(fig_resid,
                 title='📉 Residuals vs Fitted Values',
                 xaxis_title='Fitted Values (Ŷ)',
                 yaxis_title='Residuals (Y − Ŷ)')

    return fig_sc, summary, multi_content, fig_resid


@app.callback(
    Output('reg-pred-result', 'children'),
    Input('reg-pred-btn', 'n_clicks'),
    State('reg-pred-input', 'value'),
    State('reg-x', 'value'),
    State('reg-y', 'value'),
    prevent_initial_call=True,
)
def make_prediction(n_clicks, x_input, x_col, y_col):
    if x_input is None:
        return html.Span("Please enter a value for X.", style={'color': COLORS['accent2']})

    needed = list(dict.fromkeys([x_col, y_col]))
    snap   = df.groupby('location')[needed].max().reset_index(drop=True)
    snap   = snap.dropna(subset=[x_col, y_col])
    snap   = snap[(snap[x_col] > 0) & (snap[y_col] > 0)]

    X_val = snap[x_col].values
    Y_val = snap[y_col].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_val, Y_val)

    y_pred   = slope * x_input + intercept
    n        = len(X_val)
    x_mean   = X_val.mean()
    s_e      = np.sqrt(np.sum((Y_val - (slope * X_val + intercept))**2) / (n - 2))
    se_pred  = s_e * np.sqrt(1 + 1/n + (x_input - x_mean)**2 / np.sum((X_val - x_mean)**2))
    t_crit   = stats.t.ppf(0.975, df=n - 2)
    pi_lower = y_pred - t_crit * se_pred
    pi_upper = y_pred + t_crit * se_pred

    return html.Div([
        html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '12px'}, children=[
            stat_box(f"Input X ({x_col.replace('_',' ').title()})", f"{x_input:,.2f}", COLORS['accent3']),
            stat_box(f"Predicted Y ({y_col.replace('_',' ').title()})", f"{y_pred:,.2f}", COLORS['accent1']),
            stat_box("95% PI Lower", f"{pi_lower:,.2f}", COLORS['accent4']),
            stat_box("95% PI Upper", f"{pi_upper:,.2f}", COLORS['accent4']),
        ]),
        html.Div(style={
            'padding': '12px', 'background': COLORS['card2'],
            'borderRadius': '8px', 'fontSize': '13px', 'lineHeight': '1.9',
        }, children=[
            html.Strong("Result: ", style={'color': COLORS['accent1']}),
            html.Span(
                f"When {x_col.replace('_',' ')} = {x_input:,.2f}, the model predicts "
                f"{y_col.replace('_',' ')} = {y_pred:,.2f}. ",
                style={'color': COLORS['text']},
            ),
            html.Span(
                f"The 95% Prediction Interval is [{pi_lower:,.2f}, {pi_upper:,.2f}], "
                f"meaning we expect 95% of individual countries with this X value to "
                f"fall within this range.",
                style={'color': COLORS['muted']},
            ),
        ]),
    ])


# ═══════════════════════════════════════════════════════════
# TAB 6 – COMPARE COUNTRIES
# ═══════════════════════════════════════════════════════════
def layout_compare():
    defaults = [c for c in ['Pakistan', 'India', 'United States', 'Germany', 'Brazil']
                if c in countries]
    return html.Div([
        card([
            section_title("Country Comparison"),
            html.Div("Select Countries to Compare",
                     style={'color': COLORS['muted'], 'fontSize': '12px', 'marginBottom': '8px'}),
            dcc.Dropdown(
                id='cmp-countries',
                options=[{'label': c, 'value': c} for c in countries],
                value=defaults, multi=True,
                style={'background': COLORS['card2'], 'color': '#000'},
            ),
        ]),
        html.Div(style={'height': '16px'}),
        html.Div(id='cmp-content'),
    ])


@app.callback(Output('cmp-content', 'children'), Input('cmp-countries', 'value'))
def update_compare(selected):
    if not selected:
        return html.Div("Please select at least one country.",
                        style={'color': COLORS['muted']})

    snap = df[df['location'].isin(selected)].groupby('location').agg(
        total_cases=('total_cases', 'max'),
        total_deaths=('total_deaths', 'max'),
        total_vaccinations=('total_vaccinations', 'max'),
        population=('population', 'max'),
        gdp_per_capita=('gdp_per_capita', 'max'),
        life_expectancy=('life_expectancy', 'max'),
        stringency_index=('stringency_index', 'mean'),
    ).reset_index()

    snap['CFR (%)']        = (snap['total_deaths'] / snap['total_cases'].replace(0, np.nan) * 100).round(2).fillna(0)
    snap['Cases/Million']  = (snap['total_cases']  / snap['population'] * 1e6).round(0)
    snap['Deaths/Million'] = (snap['total_deaths'] / snap['population'] * 1e6).round(0)

    # ── Radar chart ───────────────────────────────────────
    radar_metrics = ['CFR (%)', 'Cases/Million', 'Deaths/Million',
                     'life_expectancy', 'stringency_index']
    fig_radar = go.Figure()
    for _, row in snap.iterrows():
        vals_r = [row[m] for m in radar_metrics]
        maxes  = [snap[m].max() for m in radar_metrics]
        vals_n = [v / mx * 100 if mx else 0 for v, mx in zip(vals_r, maxes)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_n + [vals_n[0]],
            theta=radar_metrics + [radar_metrics[0]],
            name=row['location'], mode='lines+markers',
        ))
    apply_layout(fig_radar,
                 title='🕸️ Country Comparison (Normalized 0–100 scale)',
                 polar=dict(
                     bgcolor=COLORS['card2'],
                     radialaxis=dict(visible=True, range=[0, 100],
                                    gridcolor=COLORS['border'], color=COLORS['muted']),
                     angularaxis=dict(gridcolor=COLORS['border'], color=COLORS['text']),
                 ))

    # ── Grouped bar chart ─────────────────────────────────
    fig_bars = make_subplots(rows=1, cols=3,
                              subplot_titles=['Total Cases', 'Total Deaths', 'Total Vaccinations'])
    clrs = list(CONT_COLORS.values())
    for j, m in enumerate(['total_cases', 'total_deaths', 'total_vaccinations']):
        for i, row in snap.iterrows():
            fig_bars.add_trace(
                go.Bar(x=[row['location']], y=[row[m]],
                       name=row['location'], marker_color=clrs[i % len(clrs)],
                       showlegend=(j == 0)),
                row=1, col=j + 1,
            )
    apply_layout(fig_bars, title='📊 Side-by-Side Metrics Comparison', barmode='group')

    # ── Comparison table ──────────────────────────────────
    display = snap[['location', 'total_cases', 'total_deaths', 'CFR (%)',
                     'Cases/Million', 'Deaths/Million',
                     'life_expectancy', 'gdp_per_capita', 'stringency_index']].copy()
    display.columns = ['Country', 'Total Cases', 'Total Deaths', 'CFR (%)',
                        'Cases/M', 'Deaths/M', 'Life Exp.', 'GDP/Capita', 'Avg Stringency']
    for c in ['Total Cases', 'Total Deaths', 'Cases/M', 'Deaths/M']:
        display[c] = display[c].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")

    table = dash_table.DataTable(
        data=display.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in display.columns],
        **TABLE_STYLE,
    )

    return html.Div([
        card(dcc.Graph(figure=fig_radar, config={'displayModeBar': False})),
        html.Div(style={'height': '16px'}),
        card(dcc.Graph(figure=fig_bars, config={'displayModeBar': False})),
        html.Div(style={'height': '16px'}),
        card([
            section_title("Comparison Table", COLORS['accent3']),
            table,
        ]),
    ])


# ─────────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)