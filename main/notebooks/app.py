#Import packages
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import os
import pyarrow


#Read in processed data from github
url = 'https://raw.githubusercontent.com/statzenthusiast921/ATP_Analysis/main/main/data/model_df_v2.parquet.gzip'
atp_df = pd.read_parquet(url)

matches = pd.read_csv('https://raw.githubusercontent.com/statzenthusiast921/ATP_Analysis/main/main/data/atp_matches_till_2022.csv')
matches = matches[matches['tourney_date']>=19910101]

player_choices = sorted(atp_df['player_name'].unique())
surface_choices = sorted(atp_df['surface'].unique())


#Player --> Opponent Dictionary
player_opponent_df = atp_df[['tourney_id','player_name','match_num']]
player_opponent_df = player_opponent_df.sort_values(['tourney_id', 'match_num'], ascending=[True, True])

po_pairs = pd.merge(player_opponent_df, player_opponent_df, how = 'inner', on = ['tourney_id','match_num'])
po_pairs = po_pairs[po_pairs['player_name_x'] != po_pairs['player_name_y']]  # Remove rows where a player is paired with themselves
player_opponents_dict = po_pairs.groupby('player_name_x')['player_name_y'].agg(list).to_dict()


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'color':'white',
    'backgroundColor': '#222222'

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#626ffb',
    'color': 'white',
    'padding': '6px'
}



app = dash.Dash(__name__,assets_folder=os.path.join(os.curdir,"assets"))
server = app.server
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Welcome',value='tab-1',style=tab_style, selected_style=tab_selected_style,
        children = [
            html.Div([
                html.H1(dcc.Markdown('''**Welcome to my ATP Analysis Dashboard!**''')),
                html.Br()
                   ]),   
                html.Div([
                    html.P(dcc.Markdown('''**What is the ATP?**'''))
                ],style={'text-decoration': 'underline'}),
                html.Div([
                    html.P("Blah blah blah")
                ]),
                html.Div([
                    html.P(dcc.Markdown('''**What is the purpose of this dashboard?**''')),
                ],style={'text-decoration': 'underline'}),
                html.Div([
                    html.P("Blah"),
                    html.P("Blah")
                ]),
                html.Div([
                    html.P(dcc.Markdown('''**What data is being used for this analysis?**''')),
                ],style={'text-decoration': 'underline'}),   
                html.Div([
                       html.P(["ions.", " blah"])
                ]),
                html.Div([
                    html.P(dcc.Markdown('''**What are the limitations of this data?**''')),
                ],style={'text-decoration': 'underline'}),
                html.Div(
                    children=[
                       html.P(["Nsis."])
                    ]
                )
        ]),
        dcc.Tab(label='Match History',value='tab-2',style=tab_style, selected_style=tab_selected_style,
            children = [
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose a player:')
                    ], width = 4),
                    dbc.Col([
                        dbc.Label('Choose a court surface:')
                    ], width = 4),
                    dbc.Col([
                        dbc.Label('Choose a year range:')
                    ], width = 4)
                ]),
                dbc.Row([
                    dbc.Col([
                        #----- Player filter
                        dcc.Dropdown(
                            id='dropdown0',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in player_choices],
                            value=player_choices[0]
                        )
                    ],width=4),
                    dbc.Col([
                        #----- Surface filter
                        dcc.Dropdown(
                            id='dropdown1',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in surface_choices],
                            value = surface_choices[1]
                        )
                    ],width=4),
                        #----- Year filter
                    dbc.Col([
                        dcc.RangeSlider(
                            id='range_slider',
                            min=atp_df['year'].min(),
                            max=atp_df['year'].max(),
                            step=1,
                            value=[
                                atp_df['year'].min(), 
                                atp_df['year'].max()
                            ],
                            allowCross=False,
                            pushable=1,
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks={
                                1991: '1991',
                                2000: '2000',
                                2010: '2010',
                                2020: '2020'
                            }
                        )
                    ],width=4)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='matches_table')
                    ], width = 12)
                ])
                    
            ]
       
        ),
        dcc.Tab(label='Surface Stats',value='tab-3',style=tab_style, selected_style=tab_selected_style,
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Label('Choose a player:')
                ], width = 6),
                dbc.Col([
                    dbc.Label('Choose a court surface:')
                 ], width = 6)
            ]),
            dbc.Row([
                dbc.Col([
                #----- Player filter
                    dcc.Dropdown(
                        id='dropdown2',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in player_choices],
                        value=player_choices[0]
                    )
                ],width=6),
                dbc.Col([
                 #----- Surface filter
                    dcc.Dropdown(
                        id='dropdown3',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in surface_choices],
                        value = surface_choices[1]
                    )
                ],width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='stat_timeline_chart')
                ],width = 12)
            ])
        
        ]),
        dcc.Tab(label='Head-to-Head Matchups',value='tab-4',style=tab_style, selected_style=tab_selected_style,
            children = [
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose player:')
                    ], width = 6),
                    dbc.Col([
                        dbc.Label('Choose opponent:')
                    ], width = 6),
                ]),
                dbc.Row([
                    dbc.Col([
                    #----- Player filter
                        dcc.Dropdown(
                            id='dropdown4',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in player_choices],
                            value=player_choices[0]
                        )
                    ],width=6),
                    dbc.Col([
                    #----- Surface filter
                        dcc.Dropdown(
                            id='dropdown5',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in player_choices],
                            value = player_choices[1]
                        )
                    ],width=6),
                ]),

            ]
        )

     
    ])
])


#Configure Reactivity for Tab Colors
@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 3')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab content 4')
        ])

    

@app.callback(
    Output('matches_table','children'),
    Input('dropdown0','value'),
    Input('dropdown1','value'),
    Input('range_slider','value')
)
def match_table(dd0, dd1, range_slider):

    filtered = atp_df[
        (atp_df['player_name']==dd0) &
        (atp_df['surface']==dd1) &
        (atp_df['year']>= range_slider[0]) &
        (atp_df['year']<=range_slider[1])
    ]
    #filtered = atp_df[atp_df['player_name']=='Roger Federer']
    filtered = filtered[['tourney_name','surface','tourney_date','player_age', 'rank',
                         'round','num_aces','num_dfs','serve1_in_perc',
                         'serve1_win_perc','serve2_win_perc','num_brkpts_saved',
                         'num_brkpts_faced','outcome','total_games_won',
                         'total_games_lost','game_win_perc']]

    new_df = filtered.rename(columns={
                filtered.columns[0]: "Tourney Name",
                filtered.columns[1]: "Surface",
                filtered.columns[2]: "Tourney Date",
                filtered.columns[3]: "Player Age",
                filtered.columns[4]: "Rank",
                filtered.columns[5]: "Round",
                filtered.columns[6]: "# Aces",
                filtered.columns[7]: "# Double Faults",
                filtered.columns[8]: "1st Serve In %",
                filtered.columns[9]: "1st Serve Win %",
                filtered.columns[10]: "2nd Serve Win %",
                filtered.columns[11]: "# Breakpoints Saved",
                filtered.columns[12]: "# Breakpoints Faced",
                filtered.columns[13]: "Outcome",
                filtered.columns[14]: "# Games Won",
                filtered.columns[15]: "# Games Lost",
                filtered.columns[16]: "% Games Won"
    })

    round_order = ['F','SF','QF','R16','R32','R64','R128','RR','BR','ER']

    new_df['Round'] = pd.Categorical(new_df['Round'], categories = round_order)
    new_df = new_df.sort_values(['Tourney Date','Round'], ascending=[True, False])
    new_df['% Games Won'] = new_df['% Games Won']*100
    new_df = new_df.round(1)
    new_df['% Games Won'].astype(str) + '%'
    
    return html.Div([
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in new_df.columns],
            style_data_conditional=[{
                'if': {'row_index': 'odd'},'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'},
            #filter_action='native',
            style_data={'width': '125px', 'minWidth': '125px', 'maxWidth': '125px','overflow': 'hidden','textOverflow': 'ellipsis'},
            sort_action='native',sort_mode="multi",
            page_action="native", page_current= 0,page_size= 14,                     
            data=new_df.to_dict('records'),
            style_table={'overflowX': 'auto'}


        )
    ])


@app.callback(
    Output('stat_timeline_chart','figure'),
    Input('dropdown2','value'),
    Input('dropdown3','value')
)
def stat_timeline_chart(dd2, dd3):


    filtered = atp_df[
        (atp_df['player_name']==dd2) &
        (atp_df['surface']==dd3) 
    ]


    #filtered = atp_df[atp_df['player_name']=='Roger Federer']
    filtered = filtered[['tourney_name','surface','tourney_date','player_age', 'rank',
                         'round','num_aces','num_dfs','serve1_in_perc',
                         'serve1_win_perc','serve2_win_perc','num_brkpts_saved',
                         'num_brkpts_faced','outcome','total_games_won',
                         'total_games_lost','game_win_perc']]


    line_chart_df = filtered.groupby('tourney_date').agg({
        'num_aces':'mean',
        'num_dfs':'mean'
        }).reset_index()

    line_chart_df['tourney_date'] = line_chart_df['tourney_date'].astype(int)
    line_chart_df['tourney_date'] = pd.to_datetime(line_chart_df['tourney_date'].astype(str), format='%Y%m%d')


    line_chart = px.line(
            line_chart_df, 
            x="tourney_date", 
            y="num_aces", 
            #color='MoveFromCountry',
            markers=True,
            template = 'plotly_dark',
            labels={"tourney_date": "Tourney Date",
                    "num_aces": "# Aces"
            },
            title = 'Aces'

        )
    #line_chart.update_layout(legend_title="Country")

    return line_chart

@app.callback(
    Output('dropdown5', 'options'),#-----Filters the opponent options
    Output('dropdown5', 'value'),
    Input('dropdown4', 'value') #----- Select the player
)
def set_character_options(selected_player):
    return [{'label': i, 'value': i} for i in player_opponents_dict[selected_player]], player_opponents_dict[selected_player][0]







#app.run_server(host='0.0.0.0',port='8049')

if __name__=='__main__':
	app.run_server()