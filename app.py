
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

# Load data from Google Sheets
sheet_id = "13luw-i-SML2J-_VSgcz7XsNHldbxTcCgWbHif6t4cbc"
job_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Jobdetails_2023Full"
business_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Businessesprofile_2023Full"

df1 = pd.read_csv(job_url)
df2 = pd.read_csv(business_url)

merged_df = pd.merge(df1, df2, on='employer_user_id', how='inner')
merged_df.drop_duplicates(inplace=True)
merged_df.fillna(0, inplace=True)

filtered_df = merged_df[merged_df['jse_cluster_area_x'].str.contains('Greater Yogyakarta', case=False, na=False)]

demand_supply_analysis = filtered_df.groupby(['industry']).agg({
    'job_count': 'sum',
    'apply_count': 'sum'
}).reset_index()

demand_supply_analysis['demand_supply_ratio'] = (
    demand_supply_analysis['job_count'] / demand_supply_analysis['apply_count'].replace(0, np.nan)
)
demand_supply_analysis['demand_supply_ratio'].fillna(0, inplace=True)

demand_supply_analysis['job_vacancy_rate'] = (
    (demand_supply_analysis['job_count'] - demand_supply_analysis['apply_count']) / demand_supply_analysis['job_count']
)
demand_supply_analysis['job_vacancy_rate'].fillna(0, inplace=True)

demand_supply_analysis['application_rate'] = (
    demand_supply_analysis['apply_count'] / demand_supply_analysis['job_count']
)
demand_supply_analysis['application_rate'].fillna(0, inplace=True)

role_analysis = filtered_df.groupby(['industry', 'mapped_role']).agg({
    'job_count': 'sum',
    'apply_count': 'sum'
}).reset_index()

features = demand_supply_analysis[['job_count', 'apply_count', 'demand_supply_ratio', 'job_vacancy_rate', 'application_rate']].values

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

c = 5
m = 2

cntr, u, _, _, jm, _, fpc = fuzz.cluster.cmeans(
    normalized_features.T, c=c, m=m, error=0.005, maxiter=1000, init=None, seed=42
)

demand_supply_analysis['cluster'] = np.argmax(u, axis=0)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Supply and Demand Analysis of Semi-Skilled Workers in Yogyakarta"), className="mb-4")]),
    dcc.Interval(id="interval-component", interval=1*1000, n_intervals=0),
    dbc.Row([dbc.Col(dcc.Graph(id='demand-supply-ratio-chart'), width=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id='job-vacancy-rate-chart'), width=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id='application-rate-chart'), width=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id='cluster-scatter-plot'), width=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id='sse-plot'), width=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id='role-breakdown-chart'), width=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id='membership-matrix-plot'), width=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id='cluster-centers-plot'), width=12)])
])

@app.callback(
    [Output('demand-supply-ratio-chart', 'figure'),
     Output('job-vacancy-rate-chart', 'figure'),
     Output('application-rate-chart', 'figure'),
     Output('cluster-scatter-plot', 'figure'),
     Output('sse-plot', 'figure'),
     Output('role-breakdown-chart', 'figure'),
     Output('membership-matrix-plot', 'figure'),
     Output('cluster-centers-plot', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_charts(_):
    demand_supply_fig = go.Figure()

    demand_supply_fig.add_trace(go.Bar(
        x=demand_supply_analysis['industry'],
        y=demand_supply_analysis['job_count'],
        name='Job Count (Demand)',
        marker_color='blue'
    ))

    demand_supply_fig.add_trace(go.Bar(
        x=demand_supply_analysis['industry'],
        y=demand_supply_analysis['apply_count'],
        name='Apply Count (Supply)',
        marker_color='green'
    ))

    demand_supply_fig.add_trace(go.Scatter(
        x=demand_supply_analysis['industry'],
        y=demand_supply_analysis['demand_supply_ratio'],
        name='Demand-Supply Ratio',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='red', width=3)
    ))

    demand_supply_fig.update_layout(
        title='Job Demand vs. Supply by Industry',
        xaxis_title='Industry',
        yaxis=dict(title='Job Count / Apply Count'),
        yaxis2=dict(title='Demand-Supply Ratio', overlaying='y', side='right', showgrid=False),
        barmode='group'
    )

    job_vacancy_fig = px.bar(demand_supply_analysis, x='industry', y='job_vacancy_rate', title='Job Vacancy Rate by Industry', labels={'job_vacancy_rate': 'Job Vacancy Rate (%)'}, color_discrete_sequence=['orange'])

    application_rate_fig = px.bar(demand_supply_analysis, x='industry', y='application_rate', title='Application Rate by Industry', labels={'application_rate': 'Application Rate (%)'}, color_discrete_sequence=['purple'])

    cluster_fig = px.scatter(demand_supply_analysis, x='job_count', y='apply_count', color='cluster', size='demand_supply_ratio', title='Cluster Analysis of Job Demand vs. Supply', labels={'job_count': 'Job Count (Demand)', 'apply_count': 'Apply Count (Supply)'})

    sse_fig = px.line(x=range(len(jm)), y=jm, title='Sum of Squared Errors (Fuzzy C-Means) Over Iterations', labels={'x': 'Iteration', 'y': 'SSE (Objective Function)'})

    role_chart = px.bar(role_analysis, x='mapped_role', y='job_count', color='industry', title='Job Count by Role within Each Industry', labels={'job_count': 'Job Count', 'mapped_role': 'Role'}, barmode='stack')

    membership_matrix_fig = go.Figure(data=go.Heatmap(
        z=u,
        x=[f"Cluster {i+1}" for i in range(c)],
        y=demand_supply_analysis['industry'],
        colorscale='Viridis'
    ))
    membership_matrix_fig.update_layout(
        title='Membership Matrix',
        xaxis_title='Clusters',
        yaxis_title='Industry'
    )

    cluster_centers_fig = go.Figure()
    for i in range(c):
        cluster_centers_fig.add_trace(go.Scatter(
            x=[f"Feature {j+1}" for j in range(normalized_features.shape[1])],
            y=cntr[i],
            name=f"Cluster {i+1}"
        ))
    cluster_centers_fig.update_layout(
        title='Cluster Centers',
        xaxis_title='Features',
        yaxis_title='Normalized Value'
    )

    return demand_supply_fig, job_vacancy_fig, application_rate_fig, cluster_fig, sse_fig, role_chart, membership_matrix_fig, cluster_centers_fig

if __name__ == '__main__':
    app.run_server(debug=True)
