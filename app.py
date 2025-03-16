import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from data_generator import generate_sample_data, REGION_COORDINATES
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from statsmodels.regression.linear_model import WLS
import geopandas as gpd
import statsmodels.api as sm
import math
from plotly.subplots import make_subplots
import base64
from dash.exceptions import PreventUpdate
import time
from monitoring import AppMonitor
import threading
import os

# Color palette - YlGnBu-8
COLORS = {
    'yellow_light': '#FFFFD9',
    'yellow_green': '#EDF8B1',
    'green_light': '#C7E9B4',
    'green_medium': '#7FCDBB',
    'blue_light': '#41B6C4',
    'blue_medium': '#1D91C0',
    'blue_dark': '#225EA8',
    'blue_very_dark': '#0C2C84'
}

# Metric formatting functions
def format_metric(value, metric_type):
    if metric_type == 'engagement':
        return f"{value:.2%}"
    elif metric_type == 'salary':
        return f"${value:,.2f}"
    else:  # headcount
        return f"{int(value):,}"

# Add US region mapping after the required_states definition
US_REGIONS = {
    'Northeast': ['NY', 'MA'],
    'West': ['CA', 'WA'],
    'Midwest': ['IL'],
    'South': ['TX', 'FL']
}

# Reverse mapping from state to region
STATE_TO_REGION = {state: region for region, states in US_REGIONS.items() for state in states}

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Load and filter GeoJSON data
with open('merged_shapes.geojson', 'r') as f:
    geojson_data = json.load(f)

# Filter features to only include our states and countries
required_states = ['NY', 'CA', 'IL', 'MA', 'WA', 'TX', 'FL']
required_countries = ['United Kingdom', 'Germany', 'France', 'Japan', 'Australia']

filtered_features = [
    feature for feature in geojson_data['features']
    if (feature['properties'].get('state') in required_states or
        feature['properties'].get('name') in required_countries)
]

filtered_geojson = {
    'type': 'FeatureCollection',
    'features': filtered_features
}

# Generate or load the data
df = generate_sample_data()

# Get unique values for filters
managers = ['All'] + sorted(df['reporting_manager'].unique().tolist())
metrics = ['Engagement', 'Salary', 'Headcount']

def create_waffle_chart(values, names, n_rows=5):
    """Create a waffle chart using a grid of squares."""
    n_blocks = 100  # Total number of blocks
    n_cols = math.ceil(n_blocks / n_rows)
    
    # Calculate number of blocks per category
    blocks = [round(value * n_blocks) for value in values]
    
    # Create the grid
    grid = []
    block_count = 0
    category_index = 0
    
    for row in range(n_rows):
        grid_row = []
        for col in range(n_cols):
            if block_count >= sum(blocks[:category_index + 1]):
                category_index += 1
            if category_index < len(names) and block_count < n_blocks:
                grid_row.append(category_index)
            else:
                grid_row.append(-1)  # Empty cell
            block_count += 1
        grid.append(grid_row)
    
    return grid

# Loading overlay component
loading_overlay = dbc.Modal(
    [
        dbc.ModalBody([
            html.Div([
                html.H4("Loading Dashboard...", className="mb-3"),
                dbc.Spinner(size="lg", color="primary", type="grow"),
                html.P("This may take up to 30 seconds on first load.", className="text-muted mt-3"),
            ], className="text-center")
        ])
    ],
    id="loading-overlay",
    is_open=True,
    centered=True,
    backdrop="static",
)

# App layout
app.layout = dbc.Container([
    loading_overlay,  # Add the loading overlay at the top
    # Add header row with title and logo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.Img(
                                src='data:image/png;base64,' + base64.b64encode(open('Plotly_logo.png', 'rb').read()).decode(),
                                style={'height': '60px'}  # Increased from 40px to 60px
                            ),
                            width="auto",
                            className="d-flex align-items-center"
                        ),
                        dbc.Col(
                            html.H1("Geomap Dashboard", className="mb-0 text-white"),
                            className="d-flex align-items-center"
                        )
                    ], align="center", className="px-3")  # Added padding to the row
                ], className="py-3")  # Added vertical padding to the card body
            ], style={'backgroundColor': COLORS['blue_very_dark']})  # Changed from blue_medium to blue_very_dark
        ], width=12)
    ], className="mb-4"),
    dbc.Tabs([
        dbc.Tab([
            # First tab content
            dbc.Row([
                # Left side - Geographic Distribution
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Geographic Distribution", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Manager:"),
                                    dcc.Dropdown(
                                        id='manager-filter',
                                        options=[{'label': m, 'value': m} for m in managers],
                                        value='All'
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Select Metric:"),
                                    dcc.Dropdown(
                                        id='metric-selector',
                                        options=[{'label': m, 'value': m.lower()} for m in metrics],
                                        value='engagement'
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            dcc.Graph(id='geo-heatmap', style={'height': '65vh'})
                        ])
                    ])
                ], width=6),
                
                # Right side - container for other charts
                dbc.Col([
                    # Top row - Tenure and Regional Distribution
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H4("Tenure Analysis", className="mb-0")),
                                dbc.CardBody([
                                    dcc.Graph(id='tenure-line-chart', style={'height': '30vh'})
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H4("Regional Distribution", className="mb-0")),
                                dbc.CardBody([
                                    dcc.Graph(id='us-distribution-chart', style={'height': '30vh'})
                                ])
                            ])
                        ], width=6)
                    ], className="mb-3"),
                    
                    # Bottom row - Sankey diagram
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H4("Employee Movement", className="mb-0")),
                                dbc.CardBody([
                                    dcc.Graph(id='movement-sankey', style={'height': '60vh'})  # Increased from 30vh to 45vh
                                ])
                            ])
                        ])
                    ])
                ], width=6)
            ], className="g-3")
        ], label="Employee Movement"),
        dbc.Tab([
            # Second tab content with date slider
            dbc.Row([
                dbc.Col([
                    html.Label("Select Date:", className="mb-2"),
                    html.Div([  # Added container div for better spacing
                        dcc.Slider(
                            id='date-slider',
                            min=int(pd.Timestamp(df['date'].min()).timestamp()),
                            max=int(pd.Timestamp(df['date'].max()).timestamp()),
                            value=int(pd.Timestamp(df['date'].min()).timestamp()),
                            marks={
                                int(pd.Timestamp(date).timestamp()): {
                                    'label': date.strftime('%Y-%m'),
                                    'style': {
                                        'white-space': 'nowrap',
                                        'margin-top': '30px',
                                        'font-size': '12px',
                                        'transform': 'rotate(-45deg)',
                                        'transform-origin': 'top center'
                                    }
                                }
                                for date in pd.date_range(df['date'].min(), df['date'].max(), freq='M')
                            },
                            step=None
                        )
                    ], style={'height': '120px', 'padding': '20px 0'})  # Increased height for rotated labels
                ], width=12, style={'margin-bottom': '30px'}),
            ], className="mb-4"),
            dbc.Row([
                # Left column - Footprint Analysis and Engagement Distribution
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Cost Structure Analysis", className="mb-0")),
                        dbc.CardBody([
                            dcc.Graph(id='footprint-heatmap', style={'height': '400px'})
                        ])
                    ]),
                    html.Div(className="mb-4"),  # Spacing
                    dbc.Card([
                        dbc.CardHeader(html.H4("Engagement Distribution", className="mb-0")),
                        dbc.CardBody([
                            dcc.Graph(id='engagement-distribution', style={'height': '400px'})
                        ])
                    ])
                ], width=6),
                # Right column - Geospatial Engagement and Correlation Analysis
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Geospatial Engagement", className="mb-0")),
                        dbc.CardBody([
                            dcc.Graph(id='engagement-scatter', style={'height': '400px'})
                        ])
                    ]),
                    html.Div(className="mb-4"),  # Spacing
                    dbc.Card([
                        dbc.CardHeader(html.H4("Spatial Correlation Analysis", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select X Variable:"),
                                    dcc.Dropdown(
                                        id='x-variable',
                                        options=[
                                            {'label': 'Salary', 'value': 'salary'},
                                            {'label': 'Engagement', 'value': 'engagement'},
                                            {'label': 'Headcount', 'value': 'headcount'}
                                        ],
                                        value='salary'
                                    )
                                ]),
                                dbc.Col([
                                    html.Label("Select Y Variable:"),
                                    dcc.Dropdown(
                                        id='y-variable',
                                        options=[
                                            {'label': 'Salary', 'value': 'salary'},
                                            {'label': 'Engagement', 'value': 'engagement'},
                                            {'label': 'Headcount', 'value': 'headcount'}
                                        ],
                                        value='engagement'
                                    )
                                ])
                            ], className="mb-3"),
                            dcc.Graph(id='spatial-regression', style={'height': '300px'})
                        ])
                    ])
                ], width=6)
            ])
        ], label="Location Analytics")
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output('geo-heatmap', 'figure'),
     Output('tenure-line-chart', 'figure'),
     Output('us-distribution-chart', 'figure'),
     Output('movement-sankey', 'figure')],
    [Input('manager-filter', 'value'),
     Input('metric-selector', 'value')]
)
def update_charts(manager, metric):
    # Filter data based on manager selection
    filtered_df = df if manager == 'All' else df[df['reporting_manager'] == manager]
    
    # Create metric map
    metric_map = {
        'engagement': 'engagement',
        'salary': 'salary',
        'headcount': 'employee_id'
    }
    
    # Calculate metrics for both state and country level
    agg_dict = {
        'engagement': 'mean',
        'salary': 'mean',
        'employee_id': 'count'
    }
    
    state_data = filtered_df[filtered_df['country'] == 'United States'].groupby('state').agg(agg_dict).reset_index()
    country_data = filtered_df[filtered_df['country'] != 'United States'].groupby('country').agg(agg_dict).reset_index()
    
    # Create hover text with all metrics
    def create_hover_text(row):
        location = row['state'] if 'state' in row else row['country']
        return f"<b>{location}</b><br>" + \
               f"Engagement: {format_metric(row['engagement'], 'engagement')}<br>" + \
               f"Salary: {format_metric(row['salary'], 'salary')}<br>" + \
               f"Headcount: {format_metric(row['employee_id'], 'headcount')}"
    
    state_data['hover_text'] = state_data.apply(create_hover_text, axis=1)
    country_data['hover_text'] = country_data.apply(create_hover_text, axis=1)
    
    # Create geographic heatmap
    geo_fig = go.Figure()
    
    # Add US states trace
    geo_fig.add_trace(go.Choropleth(
        locations=state_data['state'],
        z=state_data[metric_map[metric]],
        locationmode='USA-states',
        name='US States',
        colorscale=[
            [0, COLORS['yellow_light']],
            [0.2, COLORS['yellow_green']],
            [0.4, COLORS['green_light']],
            [0.6, COLORS['green_medium']],
            [0.8, COLORS['blue_medium']],
            [1, COLORS['blue_very_dark']]
        ],
        showscale=True,
        colorbar=dict(
            title=None,  # Remove side title
            len=0.5,
            thickness=10,
            ticks='outside',
            ticklen=5,
            tickfont=dict(size=10)
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=state_data['hover_text']
    ))
    
    # Add non-US countries trace
    geo_fig.add_trace(go.Choropleth(
        locations=country_data['country'],
        z=country_data[metric_map[metric]],
        locationmode='country names',
        name='Countries',
        colorscale=[
            [0, COLORS['yellow_light']],
            [0.2, COLORS['yellow_green']],
            [0.4, COLORS['green_light']],
            [0.6, COLORS['green_medium']],
            [0.8, COLORS['blue_medium']],
            [1, COLORS['blue_very_dark']]
        ],
        showscale=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=country_data['hover_text']
    ))
    
    geo_fig.update_layout(
        title=None,
        margin=dict(l=0, r=0, t=30, b=0),
        geo=dict(
            scope='world',
            showframe=False,
            showcoastlines=True,
            projection_scale=1.5,
            center=dict(lat=30, lon=0)
        ),
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='gray',
            font=dict(size=12)
        )
    )
    
    # Create tenure line chart
    tenure_data = filtered_df.groupby(['date', 'tenure_range']).agg({
        metric_map[metric]: 'mean' if metric != 'headcount' else 'count'
    }).reset_index()
    
    tenure_fig = go.Figure()
    
    for tenure in ['Low', 'Medium', 'High']:
        tenure_subset = tenure_data[tenure_data['tenure_range'] == tenure]
        
        # Format y-values based on metric type
        hover_text = [
            f"<b>{tenure}</b><br>" +
            f"{metric.title()}: {format_metric(y_val, metric.lower())}<br>" +
            f"Date: {date.strftime('%b %Y')}"
            for y_val, date in zip(tenure_subset[metric_map[metric]], tenure_subset['date'])
        ]
        
        # Update tenure line chart colors
        color = COLORS['green_light'] if tenure == 'Low' else COLORS['blue_medium'] if tenure == 'Medium' else COLORS['blue_very_dark']
        
        tenure_fig.add_trace(go.Scatter(
            x=tenure_subset['date'],
            y=tenure_subset[metric_map[metric]],
            name=tenure,
            mode='lines+markers',
            line=dict(
                width=1.5,
                color=color
            ),
            marker=dict(
                size=6,
                color=color
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_text
        ))
    
    tenure_fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title=metric.title(),
        showlegend=True,
        legend_title_text="Tenure Range",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            dtick='M1',
            tickformat='%b %Y',
            tickangle=45,
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            tickfont=dict(size=9),
            title_font=dict(size=10)
        ),
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='gray',
            font=dict(size=11)
        ),
        margin=dict(l=50, r=20, t=20, b=50),  # Reduced margins
        legend=dict(
            font=dict(size=9),
            title_font=dict(size=10)
        )
    )
    
    # Format y-axis based on metric type
    if metric == 'engagement':
        tenure_fig.update_layout(yaxis_tickformat='.1%')
    elif metric == 'salary':
        tenure_fig.update_layout(yaxis_tickformat='$,.0f')
    else:  # headcount
        tenure_fig.update_layout(yaxis_tickformat=',.0f')
    
    # Create US vs non-US distribution bar chart
    distribution_data = filtered_df.groupby(['date', 'country']).size().reset_index(name='count')
    distribution_data['region'] = distribution_data['country'].map(
        lambda x: 'US' if x == 'United States' else 'Non-US'
    )
    
    # Calculate percentages
    total_by_date = distribution_data.groupby('date')['count'].sum().reset_index()
    distribution_data = distribution_data.merge(total_by_date, on='date', suffixes=('', '_total'))
    distribution_data['percentage'] = distribution_data['count'] / distribution_data['count_total'] * 100
    
    # Group by date and region
    distribution_data = distribution_data.groupby(['date', 'region'])['percentage'].sum().reset_index()
    
    distribution_fig = go.Figure()
    
    for region in ['US', 'Non-US']:
        region_data = distribution_data[distribution_data['region'] == region]
        
        # Create hover text
        hover_text = [
            f"<b>{region}</b><br>" +
            f"Date: {date.strftime('%b %Y')}<br>" +
            f"Percentage: {percentage:.1f}%"
            for date, percentage in zip(region_data['date'], region_data['percentage'])
        ]
        
        # Update distribution bar chart colors
        color = COLORS['blue_medium'] if region == 'US' else COLORS['green_light']
        
        distribution_fig.add_trace(go.Bar(
            x=region_data['date'],
            y=region_data['percentage'],
            name=region,
            marker_color=color,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_text
        ))
    
    distribution_fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title="Percentage",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            dtick='M1',
            tickformat='%b %Y',
            tickangle=45,
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            tickformat='.0f',
            ticksuffix='%',
            range=[0, 100],
            zeroline=False,
            tickfont=dict(size=9),
            title_font=dict(size=10)
        ),
        barmode='stack',
        bargap=0.2,
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='gray',
            font=dict(size=11)
        ),
        margin=dict(l=50, r=20, t=20, b=50)  # Reduced margins
    )
    
    # Create Sankey diagram for employee movement
    # Get start and end dates
    start_date = filtered_df['date'].min()
    end_date = filtered_df['date'].max()
    
    # Get employee positions at start and end of year
    start_positions = filtered_df[filtered_df['date'] == start_date][['employee_id', 'region']].copy()
    end_positions = filtered_df[filtered_df['date'] == end_date][['employee_id', 'region']].copy()
    
    print("\nDEBUG INFO:")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print("\nSample of movements:")
    
    # Merge to track individual movements
    employee_moves = pd.merge(
        start_positions,
        end_positions,
        on='employee_id',
        suffixes=('_start', '_end')
    )
    
    print("\nFlow counts:")
    flows = employee_moves[employee_moves['region_start'] != employee_moves['region_end']].groupby(
        ['region_start', 'region_end']
    ).size().reset_index(name='value')
    print(flows)
    
    if not flows.empty:
        # Get unique source and target regions
        source_regions = sorted(flows['region_start'].unique())
        target_regions = sorted(flows['region_end'].unique())
        
        # Create nodes list with explicit source and target nodes
        nodes = []
        # Add source nodes first
        for region in source_regions:
            nodes.append({
                'label': f"{region}",
                'is_us': region != 'Non-US'
            })
        # Add target nodes
        for region in target_regions:
            nodes.append({
                'label': f"{region}",
                'is_us': region != 'Non-US'
            })
        
        # Create separate index mappings for source and target
        source_idx = {region: idx for idx, region in enumerate(source_regions)}
        target_idx = {region: idx + len(source_regions) for idx, region in enumerate(target_regions)}
        
        # Create links
        links = []
        link_colors = []
        for _, row in flows.iterrows():
            source = source_idx[row['region_start']]
            target = target_idx[row['region_end']]
            source_is_us = row['region_start'] != 'Non-US'
            target_is_us = row['region_end'] != 'Non-US'
            
            # Determine flow color with new palette
            if source_is_us and target_is_us:
                color = f"rgba({int(COLORS['blue_medium'][1:3], 16)}, {int(COLORS['blue_medium'][3:5], 16)}, {int(COLORS['blue_medium'][5:7], 16)}, 0.4)"
            elif not source_is_us and not target_is_us:
                color = f"rgba({int(COLORS['green_light'][1:3], 16)}, {int(COLORS['green_light'][3:5], 16)}, {int(COLORS['green_light'][5:7], 16)}, 0.4)"
            else:
                color = f"rgba({int(COLORS['yellow_green'][1:3], 16)}, {int(COLORS['yellow_green'][3:5], 16)}, {int(COLORS['yellow_green'][5:7], 16)}, 0.4)"
            
            links.append({
                'source': source,
                'target': target,
                'value': row['value']
            })
            link_colors.append(color)
        
        # Create Sankey diagram with updated node colors
        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=[node['label'] for node in nodes],
                color=[COLORS['blue_medium'] if node['is_us'] else COLORS['green_light'] for node in nodes],
                x=[0.05 if i < len(source_regions) else 0.95 for i in range(len(nodes))],
                y=[{
                    'Non-US': 0.1,
                    'Northeast': 0.3,
                    'Midwest': 0.5,
                    'South': 0.7,
                    'West': 0.9
                }[nodes[i]['label']] for i in range(len(nodes))]
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color=link_colors
            )
        )])
        
        sankey_fig.update_layout(
            title=None,
            font=dict(size=9),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,  # Increased from 400 to 500
            autosize=True,
            margin=dict(l=80, r=80, t=40, b=40),  # Increased margins for better spacing
            showlegend=False,
            hoverlabel=dict(
                bgcolor='white',
                bordercolor='gray',
                font=dict(size=11)
            )
        )
    else:
        # Create empty figure with a message if no movements
        sankey_fig = go.Figure()
        sankey_fig.add_annotation(
            text="No employee movements detected in the selected time period",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
    
    return geo_fig, tenure_fig, distribution_fig, sankey_fig

# Callback for footprint heatmap
@app.callback(
    Output('footprint-heatmap', 'figure'),
    Input('date-slider', 'value')
)
def update_footprint(selected_date):
    # Convert timestamp back to datetime
    date = pd.Timestamp.fromtimestamp(selected_date)
    
    # Filter data for selected date
    date_df = df[df['date'].dt.strftime('%Y-%m') == date.strftime('%Y-%m')]
    
    # Create bins for headcount and salary
    n_bins = 20  # Number of bins for each axis
    
    # Create headcount bins
    headcount_bins = np.linspace(
        date_df['employee_id'].min(),
        date_df['employee_id'].max(),
        n_bins + 1
    )
    
    # Create salary bins
    salary_bins = np.linspace(
        date_df['salary'].min(),
        date_df['salary'].max(),
        n_bins + 1
    )
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        date_df['employee_id'],
        date_df['salary'],
        bins=[headcount_bins, salary_bins],
        weights=date_df['engagement']
    )
    
    # Calculate the average engagement for each bin (avoiding division by zero)
    counts, _, _ = np.histogram2d(
        date_df['employee_id'],
        date_df['salary'],
        bins=[headcount_bins, salary_bins]
    )
    H = np.divide(H, counts, out=np.zeros_like(H), where=counts != 0)
    
    # Create heatmap
    footprint_fig = go.Figure(data=go.Heatmap(
        x=xedges[:-1],  # Use bin edges for x-axis
        y=yedges[:-1],  # Use bin edges for y-axis
        z=H.T,  # Transpose to match x and y axes
        colorscale='YlGnBu',
        showscale=True,
        colorbar=dict(title='Average Engagement'),
        hoverongaps=False,
        hovertemplate="Headcount: %{x:.0f}<br>" +
                     "Salary: $%{y:,.0f}<br>" +
                     "Engagement: %{z:.1%}<extra></extra>"
    ))
    
    footprint_fig.update_layout(
        xaxis_title='Headcount',
        yaxis_title='Salary',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,  # Changed from 600 to match container height
        autosize=True,  # Added to make it responsive
        yaxis=dict(
            gridcolor='lightgray',
            tickformat='$,.0f',
            zeroline=False,
            showgrid=True,
            automargin=True  # Added to handle tick labels better
        ),
        xaxis=dict(
            gridcolor='lightgray',
            zeroline=False,
            showgrid=False,
            automargin=True  # Added to handle tick labels better
        ),
        margin=dict(l=60, r=20, t=20, b=60)  # Added explicit margins
    )
    
    return footprint_fig

# Callback for engagement scatter
@app.callback(
    Output('engagement-scatter', 'figure'),
    Input('date-slider', 'value')
)
def update_engagement_scatter(selected_date):
    # Convert timestamp back to datetime
    date = pd.Timestamp.fromtimestamp(selected_date)
    
    # Filter data for selected date
    date_df = df[df['date'].dt.strftime('%Y-%m') == date.strftime('%Y-%m')]
    
    # Aggregate data by location
    scatter_data = date_df.groupby(['region']).agg({
        'engagement': 'mean',
        'salary': 'mean',
        'employee_id': 'count'
    }).reset_index()
    
    # Create color mapping for regions
    region_colors = {
        'Northeast': COLORS['blue_dark'],
        'West': COLORS['blue_medium'],
        'Midwest': COLORS['green_medium'],
        'South': COLORS['green_light'],
        'Non-US': COLORS['yellow_green']
    }
    
    # Create hover text manually
    hover_text = [
        f"Region: {region}<br>" +
        f"Salary: ${salary:,.0f}<br>" +
        f"Engagement: {engagement:.2%}<br>" +
        f"Headcount: {count}"
        for region, salary, engagement, count in zip(
            scatter_data['region'],
            scatter_data['salary'],
            scatter_data['engagement'],
            scatter_data['employee_id']
        )
    ]
    
    # Create bubble chart
    scatter_fig = go.Figure()
    
    # Add traces for each region
    for region in region_colors:
        region_data = scatter_data[scatter_data['region'] == region]
        if not region_data.empty:
            scatter_fig.add_trace(go.Scatter(
                x=region_data['salary'],
                y=region_data['engagement'],
                mode='markers',
                name=region,
                marker=dict(
                    size=region_data['employee_id']/2,
                    sizemode='area',
                    sizeref=2.*max(scatter_data['employee_id'])/(60.**2),
                    sizemin=30,
                    color=region_colors[region],
                    opacity=0.3,
                    line=dict(
                        color=region_colors[region],
                        width=2
                    )
                ),
                text=[hover_text[i] for i, r in enumerate(scatter_data['region']) if r == region],
                hoverinfo='text',
                showlegend=False
            ))
    
    # Add region labels to the points with better positioning
    for i, row in scatter_data.iterrows():
        scatter_fig.add_annotation(
            x=row['salary'],
            y=row['engagement'],
            text=row['region'],
            showarrow=False,
            yshift=35,
            font=dict(size=11, color='black')
        )
    
    scatter_fig.update_layout(
        xaxis_title='Average Salary',
        yaxis_title='Average Engagement',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickformat='$,.0f',
            zeroline=False,
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            tickformat='.0%',
            zeroline=False,
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        margin=dict(l=60, r=20, t=20, b=60)
    )
    
    return scatter_fig

# Callback for spatial regression
@app.callback(
    Output('spatial-regression', 'figure'),
    [Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('date-slider', 'value')]
)
def update_spatial_regression(x_var, y_var, selected_date):
    if not x_var or not y_var:
        return {}
    
    # Convert timestamp back to datetime
    date = pd.Timestamp.fromtimestamp(selected_date)
    
    # Filter data for selected date
    date_df = df[df['date'].dt.strftime('%Y-%m') == date.strftime('%Y-%m')]
    
    # Map variable names to actual columns
    var_map = {
        'headcount': 'employee_id',
        'salary': 'salary',
        'engagement': 'engagement'
    }
    
    # Get variables using the mapping
    x = date_df[var_map[x_var]].values
    y = date_df[var_map[y_var]].values
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(x, y)[0, 1]
    
    # Standardize variables
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x.reshape(-1, 1)).flatten()
    y_std = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Fit simple linear regression for the line
    model = sm.OLS(y_std, sm.add_constant(x_std)).fit()
    
    # Create regression plot
    reg_fig = go.Figure()
    
    # Add individual points
    reg_fig.add_trace(go.Scatter(
        x=x_std,
        y=y_std,
        mode='markers',
        marker=dict(
            color='rgba(31, 119, 180, 0.6)',
            size=6
        ),
        showlegend=False
    ))
    
    # Add regression line
    x_range = np.linspace(x_std.min(), x_std.max(), 100)
    y_pred = model.params[0] + model.params[1] * x_range
    
    reg_fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color='red', width=2),
        showlegend=False
    ))
    
    # Add correlation coefficient annotation
    reg_fig.add_annotation(
        x=0.95,
        y=0.95,
        xref='paper',
        yref='paper',
        text=f'r = {correlation:.2f}',
        showarrow=False,
        font=dict(size=12),
        align='right'
    )
    
    reg_fig.update_layout(
        xaxis_title=x_var.replace('_', ' ').title(),
        yaxis_title=y_var.replace('_', ' ').title(),
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        showlegend=False,
        margin=dict(l=50, r=20, t=20, b=50)
    )
    
    return reg_fig

# Replace waffle chart with horizontal bar chart
@app.callback(
    Output('engagement-distribution', 'figure'),
    Input('date-slider', 'value')
)
def update_engagement_distribution(selected_date):
    # Convert timestamp back to datetime
    date = pd.Timestamp.fromtimestamp(selected_date)
    
    # Filter data for selected date
    date_df = df[df['date'].dt.strftime('%Y-%m') == date.strftime('%Y-%m')]
    
    # Calculate engagement percentages by region
    engagement_by_region = date_df.groupby('region')['engagement'].mean()
    
    # Create subplot figure
    num_regions = len(engagement_by_region)
    num_rows = (num_regions + 2) // 3  # Ensure at least 2 rows
    num_cols = 3
    
    dist_fig = make_subplots(
        rows=num_rows, 
        cols=num_cols,
        subplot_titles=list(engagement_by_region.index),
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )
    
    # Constants for waffle layout
    n_cols = 10  # Number of columns in the grid
    n_rows = 10  # Number of rows in the grid
    total_squares = n_rows * n_cols  # Total squares per region
    marker_size = 15  # Size of markers
    
    # For each region, create a waffle chart in its subplot
    for idx, (region, engagement_pct) in enumerate(engagement_by_region.items()):
        subplot_row = (idx // num_cols) + 1
        subplot_col = (idx % num_cols) + 1
        
        n_filled = round(engagement_pct * total_squares)  # Number of filled markers
        
        # Create coordinates for all markers
        x_coords = []
        y_coords = []
        for i in range(total_squares):
            col = i % n_cols
            row = i // n_cols
            x_coords.append(col)
            y_coords.append(row)
        
        # Add background markers (light gray)
        dist_fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    symbol='square',
                    color='rgba(200,200,200,0.2)',
                    line=dict(width=1, color='rgba(200,200,200,0.5)')
                ),
                hoverinfo='skip',
                showlegend=False
            ),
            row=subplot_row,
            col=subplot_col
        )
        
        # Add filled markers
        x_coords_filled = x_coords[:n_filled]
        y_coords_filled = y_coords[:n_filled]
        
        dist_fig.add_trace(
            go.Scatter(
                x=x_coords_filled,
                y=y_coords_filled,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    symbol='square',
                    color=COLORS['blue_medium'],
                    opacity=0.7,
                    line=dict(width=1, color=COLORS['blue_medium'])
                ),
                hoverinfo='skip',
                showlegend=False
            ),
            row=subplot_row,
            col=subplot_col
        )
        
        # Add percentage annotation
        dist_fig.add_annotation(
            x=n_cols/2,
            y=-1,
            text=f"{engagement_pct:.1%}",
            showarrow=False,
            xanchor='center',
            yanchor='top',
            font=dict(
                size=14,
                color='black'
            ),
            row=subplot_row,
            col=subplot_col
        )
    
    # Update layout for all subplots
    for i in range(1, num_rows + 1):
        for j in range(1, num_cols + 1):
            dist_fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                range=[-0.5, n_cols - 0.5],
                row=i,
                col=j
            )
            dist_fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                range=[-1.5, n_rows - 0.5],
                row=i,
                col=j
            )
    
    # Update overall layout
    dist_fig.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return dist_fig

# Add callback to handle the loading state
@app.callback(
    Output("loading-overlay", "is_open"),
    [Input('geo-heatmap', 'figure'),
     Input('tenure-line-chart', 'figure'),
     Input('us-distribution-chart', 'figure'),
     Input('movement-sankey', 'figure')]
)
def hide_loading_overlay(*args):
    # Check if any of the figures are None or empty
    if any(fig is None or not fig for fig in args):
        raise PreventUpdate
    
    # Add a small delay to ensure all content is rendered
    time.sleep(0.5)
    return False

# Add before app initialization
def start_monitoring(app_url):
    monitor = AppMonitor(app_url)
    while True:
        monitor.check_status()
        monitor.log_uptime()
        time.sleep(300)  # Check every 5 minutes

# Add after app initialization but before layout
app_url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:8050')
monitoring_thread = threading.Thread(target=start_monitoring, args=(app_url,), daemon=True)

if __name__ == '__main__':
    monitoring_thread.start()
    port = int(os.environ.get('PORT', 10000))
    app.run_server(debug=False, host='0.0.0.0', port=port) 
