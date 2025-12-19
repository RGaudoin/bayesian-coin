"""
Interactive Bayesian Coin Inference App

A Dash-based web application for exploring Bayesian inference on coin fairness.
Run with: python app.py
Then open http://localhost:8050 in your browser.
"""

import sys
sys.path.insert(0, "./src")

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bayesian_coin import BetaPrior, DeltaPrior, CompoundPrior
from utils import mk_coin


# Global state for the simulation
class SimulationState:
    def __init__(self):
        self.reset()

    def reset(self, p_0=1, p_1=1, p_fair=1000000, p_noninf=0.001, combine=True,
              beta_alpha=0.5, beta_beta=0.5):
        self.coin = mk_coin(p_0, p_1, p_fair, p_noninf, combine=combine,
                            beta_alpha=beta_alpha, beta_beta=beta_beta)
        self.combine = combine
        self.weights_history = []
        self.flips_history = []
        self.stats_history = []
        self.true_prob_history = []
        self.step = 0

    def run_steps(self, n_steps, true_prob):
        for _ in range(n_steps):
            flip = np.random.binomial(1, true_prob, size=1)
            self.coin.update(flip)
            self.weights_history.append(self.coin._params['logweights'].copy())
            self.flips_history.append(flip[0])
            self.stats_history.append(self.coin.stats(std=True))
            self.true_prob_history.append(true_prob)
            self.step += 1

    def get_dataframes(self):
        if not self.weights_history:
            return None, None, None

        # Column names depend on whether priors are combined
        if self.combine:
            weight_columns = ['One-sided', 'Fair', 'Unknown']
        else:
            weight_columns = ['Always Tails', 'Always Heads', 'Fair', 'Unknown']

        df_weights = pd.DataFrame(np.array(self.weights_history),
                                   columns=weight_columns)
        df_stats = pd.DataFrame(np.array(self.stats_history),
                                 columns=['Mean', 'Std'])
        df_flips = pd.DataFrame({
            'flip': self.flips_history,
            'true_prob': self.true_prob_history,
            'cumulative_mean': pd.Series(self.flips_history).expanding().mean()
        })
        return df_weights, df_stats, df_flips


state = SimulationState()

# Create the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# MathJax configuration for LaTeX rendering
mathjax_script = html.Script(
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
)

# Theory/Formulas content
theory_content = dcc.Markdown('''
## Mathematical Framework

### Core Idea: Composable Priors

The framework implements a **composable prior hierarchy** where any prior can contain sub-priors,
which can themselves be mixtures. This recursive structure is not limited to the binomial/coin-flip
case—it generalizes to any likelihood function with compatible priors.

```
CompoundPrior (mixture)
├── DeltaPrior (p=0)          # Always tails
├── DeltaPrior (p=1)          # Always heads
├── DeltaPrior (p=0.5)        # Fair coin
├── BetaPrior (α, β)          # Unknown bias
└── CompoundPrior (nested)    # Sub-mixtures allowed!
```

**Impossible priors are acceptable:** When a hypothesis becomes impossible (e.g., p=0 after seeing heads),
the mixture update simply sets its weight to zero, excluding it from further inference.

---

### 1. Beta Prior (Conjugate to Bernoulli)

**Conjugate Update** (after observing k heads, m tails):

> α' = α₀ + k,  β' = β₀ + m

**Mean:**  E[p] = α / (α + β)

**Variance:**  Var(p) = αβ / [(α+β)²(α+β+1)]

**Log-Normalization:**  ln P(D|Beta) = ln B(α', β') - ln B(α₀, β₀)

| Prior Name | Parameters | Interpretation |
|------------|------------|----------------|
| Jeffreys | α=0.5, β=0.5 | Non-informative (default) |
| Uniform | α=1, β=1 | All biases equally likely |
| Haldane | α→0, β→0 | Improper, data-dominated |

---

### 2. Delta Prior (Point Mass)

Models fixed hypotheses where the coin bias is known exactly.

**Key insight:** Unlike Beta priors, Delta priors **do not update their parameters**.
The value p₀ represents a firm belief that the probability IS exactly that value, regardless of evidence.
Only the likelihood (lognorm) changes, affecting the prior's weight in a mixture.

**Log-Likelihood:**  ln P(D|p₀) = k·ln(p₀) + m·ln(1-p₀)

| Hypothesis | p₀ | Becomes impossible when... |
|------------|-----|----------------------------|
| Always tails | 0 | Any heads observed |
| Always heads | 1 | Any tails observed |
| Fair coin | 0.5 | Never |

---

### 3. Compound Prior (Mixture)

**Prior:**  P(p) = Σᵢ wᵢ Pᵢ(p)

**Weight Update (Bayes' Rule):**

> wᵢ' = P(D|Mᵢ) · wᵢ⁽⁰⁾ / Σⱼ P(D|Mⱼ) · wⱼ⁽⁰⁾

In log-space: ln wᵢ' = ln P(D|Mᵢ) + ln wᵢ⁽⁰⁾ - logsumexp(...)

**How different prior types update:**

| Prior Type | On Evidence |
|------------|-------------|
| BetaPrior | Parameters update (α += heads, β += tails). Distribution sharpens. |
| DeltaPrior | **Parameters fixed.** Only lognorm changes → affects mixture weight. |
| CompoundPrior | Recursively updates sub-priors, recomputes weights. |

**The essence of Bayesian model comparison:** The Delta prior hypothesis doesn't "learn"—it either
gets supported or ruled out by evidence. Beta priors adapt *within* a model; Delta priors compete
*between* fixed models.

**Mixture Mean:**  E[p] = Σᵢ wᵢ μᵢ

**Mixture Variance (Law of Total Variance):**

> Var(p) = Σᵢ wᵢ σᵢ² + [Σᵢ wᵢ μᵢ² - (Σᵢ wᵢ μᵢ)²]

---

### 4. Hierarchical Composability

Sub-priors can themselves be CompoundPriors:

```
Coin (CompoundPrior)
├── One-sided (CompoundPrior)
│   ├── Always tails (DeltaPrior p=0)
│   └── Always heads (DeltaPrior p=1)
├── Fair (DeltaPrior p=0.5)
└── Unknown (BetaPrior α=0.5, β=0.5)
```

**Generality:** Replace BetaPrior with any conjugate prior (Normal-Normal, Gamma-Poisson, etc.)
and the CompoundPrior mechanics remain identical.

---

### 5. Sampling

**From Beta:**  p ~ Beta(α, β)

**From Mixture:**
1. Select component i with probability wᵢ
2. Sample p ~ Pᵢ
''', style={'padding': '20px', 'maxWidth': '900px', 'margin': '0 auto',
            'backgroundColor': 'white', 'borderRadius': '10px'})

# Simulation tab content (the original layout)
simulation_content = html.Div([
    html.Div([
        # Left panel - Controls
        html.Div([
            html.H3("Prior Configuration", style={'color': '#34495e'}),

            html.Div([
                html.Label("P(always tails) weight:"),
                dcc.Input(id='p-0', type='text', value='1',
                         debounce=True,
                         style={'width': '100%', 'marginBottom': '10px'}),
            ]),

            html.Div([
                html.Label("P(always heads) weight:"),
                dcc.Input(id='p-1', type='text', value='1',
                         debounce=True,
                         style={'width': '100%', 'marginBottom': '10px'}),
            ]),

            html.Div([
                html.Label("P(fair coin) weight:"),
                dcc.Input(id='p-fair', type='text', value='1000000',
                         debounce=True,
                         style={'width': '100%', 'marginBottom': '10px'}),
            ]),

            html.Div([
                html.Label("P(unknown bias) weight:"),
                dcc.Input(id='p-noninf', type='text', value='0.001',
                         debounce=True,
                         style={'width': '100%', 'marginBottom': '10px'}),
            ]),

            html.Div([
                html.Label("Beta prior parameters (for unknown bias):"),
                html.Div([
                    html.Div([
                        html.Label("α:", style={'fontSize': '12px'}),
                        dcc.Input(id='beta-alpha', type='text', value='0.5',
                                 debounce=True,
                                 style={'width': '100%'}),
                    ], style={'flex': '1', 'marginRight': '10px'}),
                    html.Div([
                        html.Label("β:", style={'fontSize': '12px'}),
                        dcc.Input(id='beta-beta', type='text', value='0.5',
                                 debounce=True,
                                 style={'width': '100%'}),
                    ], style={'flex': '1'}),
                ], style={'display': 'flex', 'marginBottom': '10px'}),
            ]),

            html.Div([
                dcc.Checklist(
                    id='combine-checkbox',
                    options=[{'label': ' Combine one-sided priors', 'value': 'combine'}],
                    value=['combine'],
                    style={'marginBottom': '15px'}
                ),
            ]),

            html.Button('Reset / Initialize', id='reset-btn', n_clicks=0,
                       style={'width': '100%', 'padding': '10px', 'marginBottom': '20px',
                              'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                              'borderRadius': '5px', 'cursor': 'pointer'}),

            html.Hr(),

            html.H3("Simulation Controls", style={'color': '#34495e'}),

            # Hierarchy visualization
            html.Div(id='hierarchy-diagram', style={'marginBottom': '15px'}),

            html.Div([
                html.Label("True coin probability:"),
                html.Div([
                    html.Div([
                        dcc.Slider(id='true-prob-slider', min=0, max=1, step=0.05, value=0.5,
                                  marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                  tooltip={'placement': 'bottom', 'always_visible': False}),
                    ], style={'flex': '1', 'marginRight': '10px'}),
                    dcc.Input(id='true-prob-input', type='text', value='0.5',
                             debounce=True,
                             style={'width': '60px', 'textAlign': 'center'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Number of steps:"),
                dcc.Input(id='n-steps', type='text', value='100',
                         debounce=True,
                         style={'width': '100%', 'marginBottom': '10px'}),
            ]),

            html.Button('Run Steps', id='run-btn', n_clicks=0,
                       style={'width': '100%', 'padding': '10px', 'marginBottom': '20px',
                              'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                              'borderRadius': '5px', 'cursor': 'pointer'}),

            html.Hr(),

            html.H3("Display Options", style={'color': '#34495e'}),

            dcc.Checklist(
                id='log-scale',
                options=[{'label': ' Log scale for weights', 'value': 'log'}],
                value=['log'],
                style={'marginBottom': '10px'}
            ),

            dcc.Checklist(
                id='show-probs',
                options=[{'label': ' Show as probabilities (exp)', 'value': 'probs'}],
                value=[],
                style={'marginBottom': '20px'}
            ),

            html.Hr(),

            html.Div(id='current-stats', style={'marginTop': '20px'}),

        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#ecf0f1',
                  'borderRadius': '10px', 'marginRight': '20px'}),

        # Right panel - Plots
        html.Div([
            dcc.Graph(id='weights-plot', style={'height': '400px'}),
            dcc.Graph(id='stats-plot', style={'height': '300px'}),
            dcc.Graph(id='flips-plot', style={'height': '250px'}),
        ], style={'width': '75%'}),

    ], style={'display': 'flex', 'padding': '20px'}),

    # Store for tracking button clicks
    dcc.Store(id='reset-store', data=0),
    dcc.Store(id='run-store', data=0),
])

# Main app layout with tabs
app.layout = html.Div([
    html.H1("Bayesian Coin Inference Explorer",
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),

    dcc.Tabs(id='main-tabs', value='simulation', children=[
        dcc.Tab(label='Simulation', value='simulation', children=[simulation_content]),
        dcc.Tab(label='Theory', value='theory', children=[theory_content]),
    ], style={'marginBottom': '20px'}),

], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f6fa'})


def safe_float(val, default):
    """Safely convert value to float, returning default on failure."""
    try:
        if val is not None and str(val).strip() != '':
            return float(val)
    except (ValueError, TypeError):
        pass
    return default


def make_hierarchy_diagram(combine):
    """Create a visual tree diagram of the prior hierarchy."""
    node_style = {
        'padding': '4px 8px',
        'borderRadius': '4px',
        'fontSize': '11px',
        'display': 'inline-block',
        'margin': '2px',
    }

    root_style = {**node_style, 'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'}
    branch_style = {**node_style, 'backgroundColor': '#e74c3c', 'color': 'white'}
    leaf_style = {**node_style, 'backgroundColor': '#ecf0f1', 'color': '#2c3e50', 'border': '1px solid #bdc3c7'}

    if combine:
        # Combined hierarchy: Coin -> [One-sided -> [p=0, p=1], Fair, Unknown]
        return html.Div([
            html.Div([
                html.Span("Coin", style=root_style),
            ], style={'textAlign': 'center', 'marginBottom': '5px'}),
            html.Div("├─────────┼─────────┤", style={'textAlign': 'center', 'fontFamily': 'monospace', 'color': '#7f8c8d', 'fontSize': '10px'}),
            html.Div([
                html.Span("One-sided", style=branch_style),
                html.Span(" ", style={'width': '20px', 'display': 'inline-block'}),
                html.Span("Fair (p=0.5)", style=leaf_style),
                html.Span(" ", style={'width': '20px', 'display': 'inline-block'}),
                html.Span("Unknown (Beta)", style=leaf_style),
            ], style={'textAlign': 'center', 'marginBottom': '5px'}),
            html.Div([
                html.Span("┌───┴───┐", style={'fontFamily': 'monospace', 'color': '#7f8c8d', 'fontSize': '10px'}),
            ], style={'textAlign': 'left', 'marginLeft': '25px'}),
            html.Div([
                html.Span("p=0", style=leaf_style),
                html.Span(" ", style={'width': '10px', 'display': 'inline-block'}),
                html.Span("p=1", style=leaf_style),
            ], style={'textAlign': 'left', 'marginLeft': '15px'}),
        ], style={'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'})
    else:
        # Flat hierarchy: Coin -> [p=0, p=1, Fair, Unknown]
        return html.Div([
            html.Div([
                html.Span("Coin", style=root_style),
            ], style={'textAlign': 'center', 'marginBottom': '5px'}),
            html.Div("├────┬────┬────┤", style={'textAlign': 'center', 'fontFamily': 'monospace', 'color': '#7f8c8d', 'fontSize': '10px'}),
            html.Div([
                html.Span("p=0", style=leaf_style),
                html.Span(" ", style={'width': '5px', 'display': 'inline-block'}),
                html.Span("p=1", style=leaf_style),
                html.Span(" ", style={'width': '5px', 'display': 'inline-block'}),
                html.Span("Fair", style=leaf_style),
                html.Span(" ", style={'width': '5px', 'display': 'inline-block'}),
                html.Span("Beta", style=leaf_style),
            ], style={'textAlign': 'center'}),
        ], style={'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'})


@callback(
    Output('hierarchy-diagram', 'children'),
    Input('combine-checkbox', 'value'),
)
def update_hierarchy(combine):
    combine_bool = 'combine' in (combine or [])
    return make_hierarchy_diagram(combine_bool)


@callback(
    Output('true-prob-input', 'value'),
    Output('true-prob-slider', 'value'),
    Input('true-prob-input', 'value'),
    Input('true-prob-slider', 'value'),
    prevent_initial_call=True
)
def sync_prob_controls(input_value, slider_value):
    """Sync slider and text input without circular dependency."""
    triggered_id = ctx.triggered_id
    if triggered_id == 'true-prob-slider':
        # Slider changed - update the text input
        return str(slider_value), slider_value
    else:
        # Text input changed - update the slider
        try:
            val = float(input_value)
            val = max(0, min(1, val))
            return input_value, val
        except (ValueError, TypeError):
            return input_value, 0.5


@callback(
    Output('reset-store', 'data'),
    Input('reset-btn', 'n_clicks'),
    State('p-0', 'value'),
    State('p-1', 'value'),
    State('p-fair', 'value'),
    State('p-noninf', 'value'),
    State('combine-checkbox', 'value'),
    State('beta-alpha', 'value'),
    State('beta-beta', 'value'),
    prevent_initial_call=True
)
def reset_simulation(n_clicks, p_0, p_1, p_fair, p_noninf, combine, beta_alpha, beta_beta):
    combine_bool = 'combine' in (combine or [])
    # Convert to float, handling None, empty, or invalid values
    p_0_val = safe_float(p_0, 1.0)
    p_1_val = safe_float(p_1, 1.0)
    p_fair_val = safe_float(p_fair, 1000000.0)
    p_noninf_val = safe_float(p_noninf, 0.001)
    beta_alpha_val = safe_float(beta_alpha, 0.5)
    beta_beta_val = safe_float(beta_beta, 0.5)
    print(f"Resetting with: p_0={p_0_val}, p_1={p_1_val}, p_fair={p_fair_val}, p_noninf={p_noninf_val}, combine={combine_bool}, beta_alpha={beta_alpha_val}, beta_beta={beta_beta_val}")
    state.reset(p_0_val, p_1_val, p_fair_val, p_noninf_val, combine_bool,
                beta_alpha_val, beta_beta_val)
    return n_clicks


@callback(
    Output('run-store', 'data'),
    Input('run-btn', 'n_clicks'),
    State('n-steps', 'value'),
    State('true-prob-input', 'value'),
    prevent_initial_call=True
)
def run_simulation(n_clicks, n_steps, true_prob):
    # Convert to int, handling None, empty, or invalid values
    try:
        n_steps_val = int(n_steps) if n_steps is not None and str(n_steps).strip() != '' else 100
    except (ValueError, TypeError):
        n_steps_val = 100
    # Use the text input value for precise probability
    true_prob_val = safe_float(true_prob, 0.5)
    # Clamp to valid range
    true_prob_val = max(0.0, min(1.0, true_prob_val))
    if n_steps_val > 0:
        print(f"Running {n_steps_val} steps with true_prob={true_prob_val}")
        state.run_steps(n_steps_val, true_prob_val)
    return n_clicks


@callback(
    Output('weights-plot', 'figure'),
    Output('stats-plot', 'figure'),
    Output('flips-plot', 'figure'),
    Output('current-stats', 'children'),
    Input('reset-store', 'data'),
    Input('run-store', 'data'),
    Input('log-scale', 'value'),
    Input('show-probs', 'value'),
)
def update_plots(reset_data, run_data, log_scale, show_probs):
    df_weights, df_stats, df_flips = state.get_dataframes()

    use_log = 'log' in (log_scale or [])
    use_probs = 'probs' in (show_probs or [])

    # Weights plot
    fig_weights = go.Figure()

    if df_weights is not None and len(df_weights) > 0:
        plot_data = df_weights.copy()

        if use_probs:
            plot_data = np.exp(plot_data)
            y_title = "Probability"
        else:
            y_title = "Log-weight" if use_log else "Weight"
            if not use_log:
                plot_data = np.exp(plot_data)

        colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71']
        for i, col in enumerate(plot_data.columns):
            fig_weights.add_trace(go.Scatter(
                x=list(range(len(plot_data))),
                y=plot_data[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

    fig_weights.update_layout(
        title='Hypothesis Weights Over Time',
        xaxis_title='Step',
        yaxis_title=y_title if df_weights is not None else 'Weight',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=60, b=40)
    )

    # Stats plot
    fig_stats = go.Figure()

    if df_stats is not None and len(df_stats) > 0:
        fig_stats.add_trace(go.Scatter(
            x=list(range(len(df_stats))),
            y=df_stats['Mean'],
            mode='lines',
            name='Posterior Mean',
            line=dict(color='#9b59b6', width=2)
        ))

        # Add confidence band (mean +/- std)
        fig_stats.add_trace(go.Scatter(
            x=list(range(len(df_stats))) + list(range(len(df_stats)))[::-1],
            y=list(df_stats['Mean'] + df_stats['Std']) + list(df_stats['Mean'] - df_stats['Std'])[::-1],
            fill='toself',
            fillcolor='rgba(155, 89, 182, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Mean +/- Std',
            showlegend=True
        ))

        # Add true probability line
        if df_flips is not None:
            fig_stats.add_trace(go.Scatter(
                x=list(range(len(df_flips))),
                y=df_flips['true_prob'],
                mode='lines',
                name='True Probability',
                line=dict(color='#e67e22', width=2, dash='dash')
            ))

    fig_stats.update_layout(
        title='Posterior Statistics',
        xaxis_title='Step',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=60, b=40)
    )

    # Flips plot
    fig_flips = go.Figure()

    if df_flips is not None and len(df_flips) > 0:
        fig_flips.add_trace(go.Scatter(
            x=list(range(len(df_flips))),
            y=df_flips['cumulative_mean'],
            mode='lines',
            name='Cumulative Mean',
            line=dict(color='#1abc9c', width=2)
        ))

        fig_flips.add_trace(go.Scatter(
            x=list(range(len(df_flips))),
            y=df_flips['true_prob'],
            mode='lines',
            name='True Probability',
            line=dict(color='#e67e22', width=2, dash='dash')
        ))

    fig_flips.update_layout(
        title='Observed Flips (Cumulative Mean)',
        xaxis_title='Step',
        yaxis_title='Proportion Heads',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=60, b=40)
    )

    # Current stats display
    if state.step > 0:
        mean, std = state.coin.stats(std=True)
        weights = np.exp(state.coin._params['logweights'])

        # Build hypothesis probability display based on combine mode
        if state.combine:
            hyp_probs = [
                html.P(f"One-sided: {weights[0]:.6f}"),
                html.P(f"Fair: {weights[1]:.6f}"),
                html.P(f"Unknown: {weights[2]:.6f}"),
            ]
        else:
            hyp_probs = [
                html.P(f"Always Tails: {weights[0]:.6f}"),
                html.P(f"Always Heads: {weights[1]:.6f}"),
                html.P(f"Fair: {weights[2]:.6f}"),
                html.P(f"Unknown: {weights[3]:.6f}"),
            ]

        stats_display = html.Div([
            html.H4("Current State", style={'color': '#2c3e50', 'marginBottom': '10px'}),
            html.P(f"Total steps: {state.step}"),
            html.P(f"Posterior mean: {mean:.4f}"),
            html.P(f"Posterior std: {std:.4f}"),
            html.Hr(),
            html.P(html.Strong("Hypothesis probabilities:")),
            *hyp_probs,
            html.Hr(),
            html.P(html.Strong("Beta prior params:")),
            html.P(f"alpha: {state.coin._params['priors'][-1]._params['alpha']:.1f}"),
            html.P(f"beta: {state.coin._params['priors'][-1]._params['beta']:.1f}"),
        ])
    else:
        stats_display = html.Div([
            html.P("Click 'Reset / Initialize' then 'Run Steps' to start the simulation.",
                  style={'fontStyle': 'italic', 'color': '#7f8c8d'})
        ])

    return fig_weights, fig_stats, fig_flips, stats_display


if __name__ == '__main__':
    print("Starting Bayesian Coin Inference App...")
    print("Open http://localhost:8050 in your browser")
    app.run(debug=True, host='localhost', port=8050)
