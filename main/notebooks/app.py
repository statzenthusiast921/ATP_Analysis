#Import packages
import pandas as pd
#import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import os
import pyarrow


#Read in processed data from github
url = 'https://raw.githubusercontent.com/statzenthusiast921/ATP_Analysis/main/main/data/model_df_v2.parquet.gzip'
atp_df = pd.read_parquet(url)


#Filter out players with less than 200 matches
match_totals = atp_df.groupby('player_name').agg(total_match_count=('player_name', 'size')).reset_index()

atp_df = pd.merge(
    atp_df, 
    match_totals, 
    how = 'inner', 
    on = 'player_name'
)

atp_df = atp_df[atp_df['total_match_count']>=300]

#Define options for dropdown menus
player_choices = sorted(atp_df['player_name'].unique())
surface_choices = sorted(atp_df['surface'].unique())
statistic_choices = sorted([
    'Aces','Double Faults','Break Points Saved',
    'Break Points Faced','% Games Won',
    '1st Serve In %', '1st Serve Win %','2nd Serve Win %'
])

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
                    html.P("The Association of Tennis Professionals (ATP) is the governing body of the men's professional tennis circuits - the ATP Tour, the ATP Challenger Tour and the ATP Champions Tour. It was formed in September 1972 to protect the interests of professional tennis players.")
                ]),
                html.Div([
                    html.P(dcc.Markdown('''**What is the purpose of this dashboard?**''')),
                ],style={'text-decoration': 'underline'}),
                html.Div([
                    html.P("The purpose of this dashboard is to analyze tennis match statistics to answer the following questions:"),
                    html.P("1.) How does a player's individual performance change over time?"),
                    html.P("2.) How do players head-to-head performances compare?"),
                    html.P('3.) Can we predict the outcome of a match given certain attributes?')
                ]),
                html.Div([
                    html.P(dcc.Markdown('''**What data is being used for this analysis?**''')),
                ],style={'text-decoration': 'underline'}),   
                html.Div([
                       html.P(["The data chosen for this analysis was found from this Kaggle link ", html.A('here.', href = 'https://www.kaggle.com/datasets/sijovm/atpdata/data'), " This data contains details of ATP matches since 1968."])
                ]),
                html.Div([
                    html.P(dcc.Markdown('''**What are the limitations of this data?**''')),
                ],style={'text-decoration': 'underline'}),
                html.Div(
                    children=[
                       html.P("This data only includes match statistics from 1991 to the present.  Data from 1968 through 1990 was not included due to this issue.  Further, the ATP and WTA (Women's Tennis Association) are mutually exclusive.  This dataset only contains details on ATP matches. ")
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
        dcc.Tab(label='Individual Stats',value='tab-3',style=tab_style, selected_style=tab_selected_style,
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Label('Choose a player:')
                ], width = 6),
                dbc.Col([
                    dbc.Label('Choose a statistic:')
                 ], width = 6),
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
                 #----- Statistic filter
                    dcc.Dropdown(
                        id='dropdown3',
                        style={'color':'black'},
                        options=[{'label': i, 'value': i} for i in statistic_choices],
                        value = statistic_choices[0]
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
                            value = 'Roger Federer'
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
                #----- Head to Head Stat Cards
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id="card1")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card2")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card3")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card4")
                    ],width=3)
                ],className="g-0"),
             

            ]
        ),
        dcc.Tab(label='Predict Winners',value='tab-5',style=tab_style, selected_style=tab_selected_style,
            children=[

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
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Tab content 5')
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

    filtered = atp_df[atp_df['player_name']==dd2] 

    #filtered = atp_df[atp_df['player_name']=='Roger Federer']

    filtered = filtered[['tourney_name','surface','tourney_date','player_age', 
                        'rank','num_aces','num_dfs','serve1_in_perc',
                         'serve1_win_perc','serve2_win_perc','num_brkpts_saved',
                         'num_brkpts_faced','outcome','game_win_perc']]

    filtered['game_win_perc'] = filtered['game_win_perc']*100


    stats_df = filtered.groupby('tourney_date').agg({
        'num_aces':'sum',
        'num_dfs':'sum',
        'serve1_in_perc':'mean',
        'serve1_win_perc':'mean',
        'serve2_win_perc':'mean',
        'game_win_perc': 'mean',
        'num_brkpts_faced':'sum',
        'num_brkpts_saved':'sum'
    }).reset_index()

    stats_df['tourney_date'] = pd.to_datetime(stats_df['tourney_date'], format='%Y%m%d')  
    
    first_match = stats_df['tourney_date'].min()
    last_match = stats_df['tourney_date'].max()

    delta = last_match - first_match
    num_days = delta.days


    my_range = pd.date_range(
        min(stats_df['tourney_date']), 
        periods=num_days, 
        freq='1d'
    )

    date_df = pd.DataFrame(my_range)
    date_df = date_df.rename(columns={date_df.columns[0]: "tourney_date"})

    full_days_df = pd.merge(
        date_df,
        stats_df,
        on = 'tourney_date',
        how = 'left'
    )
    #full_days_df['num_aces'] = full_days_df['num_aces'].fillna(0)
    #full_days_df['num_dfs'] = full_days_df['num_dfs'].fillna(0)
    #full_days_df['num_brkpts_faced'] = full_days_df['num_brkpts_faced'].fillna(0)
    #full_days_df['num_brkpts_saved'] = full_days_df['num_brkpts_saved'].fillna(0)

    full_days_df['year'] = full_days_df['tourney_date'].astype(str).str[0:4]
    full_days_df['month'] = full_days_df['tourney_date'].astype(str).str[5:7]
    full_days_df['ym'] = full_days_df['year'].astype(str) + "-" + full_days_df['month']

    line_chart_df = full_days_df.groupby('ym').agg({
        'num_aces':'sum',
        'num_dfs':'sum',
        'serve1_in_perc':'mean',
        'serve1_win_perc':'mean',
        'serve2_win_perc':'mean',
        'game_win_perc': 'mean',
        'num_brkpts_faced':'sum',
        'num_brkpts_saved':'sum'

    }).reset_index()

    #----- Stat #1: % Games Won
    if statistic_choices[0] in dd3:

        line_chart = px.line(
                line_chart_df, 
                x="ym", 
                y="game_win_perc", 
                markers=True,
                template = 'plotly_dark',
                labels={"ym": "Month-Year",
                        "game_win_perc": "% Games Won"
                },
                title = '% Games Won'
            )
        return line_chart

    #----- Stat #2: 1st Serve in %
    elif statistic_choices[1] in dd3:

        line_chart = px.line(
                line_chart_df, 
                x="ym", 
                y="serve1_in_perc", 
                markers=True,
                template = 'plotly_dark',
                labels={"ym": "Month-Year",
                        "serve1_in_perc": "1st Serve in %"
                },
                title = '1st Serve In %'
            )

        return line_chart

    #----- Stat #3: 1st Serve Win %
    elif statistic_choices[2] in dd3:
    
        line_chart = px.line(
                line_chart_df, 
                x="ym", 
                y="serve1_win_perc", 
                markers=True,
                template = 'plotly_dark',
                labels={"ym": "Month-Year",
                        "serve1_win_perc": "1st Serve Win %"
                },
                title = '1st Serve Win %'
            )

        return line_chart

    #----- Stat #4: 2nd Serve Win %
    elif statistic_choices[3] in dd3:
        
        line_chart = px.line(
                line_chart_df, 
                x="ym", 
                y="serve2_win_perc", 
                markers=True,
                template = 'plotly_dark',
                labels={"ym": "Month-Year",
                        "serve2_win_perc": "2nd Serve Win %"
                },
                title = '2nd Serve Win %'
            )

        return line_chart

    #----- Stat #5: Aces
    elif statistic_choices[4] in dd3:
            
        line_chart = px.line(
                line_chart_df, 
                x="ym", 
                y="num_aces", 
                markers=True,
                template = 'plotly_dark',
                labels={"ym": "Month-Year",
                        "num_aces": "# Aces"
                },
                title = '# Aces'
            )

        return line_chart

    #----- Stat #6: Break Points Faced
    elif statistic_choices[5] in dd3:
            
        line_chart = px.line(
                line_chart_df, 
                x="ym", 
                y="num_brkpts_faced", 
                markers=True,
                template = 'plotly_dark',
                labels={"ym": "Month-Year",
                        "num_brkpts_faced": "# Break Points Faced"
                },
                title = '# Break Points Faced'
            )

        return line_chart

    #----- Stat #7: Break Points Saved
    elif statistic_choices[6] in dd3:
            
        line_chart = px.line(
                line_chart_df, 
                x="ym", 
                y="num_brkpts_saved", 
                markers=True,
                template = 'plotly_dark',
                labels={"ym": "Month-Year",
                        "num_brkpts_saved": "# Break Points Saved"
                },
                title = '# Break Points Saved'
            )

        return line_chart



    #----- Stat #8: # Double Faults
    elif statistic_choices[7] in dd3:
            
        line_chart = px.line(
                line_chart_df, 
                x="ym", 
                y="num_dfs", 
                markers=True,
                template = 'plotly_dark',
                labels={"ym": "Month-Year",
                        "num_dfs": "# Double Faults"
                },
                title = '# Double Faults'
            )

        return line_chart



@app.callback(
    Output('dropdown5', 'options'),#-----Filters the opponent options
    Output('dropdown5', 'value'),
    Input('dropdown4', 'value') #----- Select the player
)
def set_character_options(selected_player):
    return [{'label': i, 'value': i} for i in player_opponents_dict[selected_player]], player_opponents_dict[selected_player][0]


@app.callback(
    Output('card1', 'children'),
    Output('card2', 'children'),
    Output('card3', 'children'),
    Output('card4', 'children'),
    Input('dropdown4', 'value'),
    Input('dropdown5', 'value')
)


def head_to_head_match_stats(dd4, dd5):
    player1_df = atp_df[atp_df['player_name']==dd4]
    player2_df = atp_df[atp_df['player_name']==dd5]

    new_df = pd.merge(
        player1_df,
        player2_df,
        how = 'inner',
        on = ['tourney_id','match_num']
    )
    #new_df.to_csv('head2head.csv', sep=',', index=False, encoding='utf-8')

    win_df = new_df[new_df['outcome_x']==1]
    loss_df = new_df[new_df['outcome_x']==0]

    wins = win_df.shape[0]
    losses = loss_df.shape[0]
    
    card1 = dbc.Card([
            dbc.CardBody([
                html.H5(f'{wins} - {losses}'),
                html.P('Wins - Losses')
            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)


    avg_aces = round(new_df['num_aces_x'].mean(),0)

    card2 = dbc.Card([
            dbc.CardBody([
                html.H5(f'{avg_aces}'),
                html.P('Avg # Aces Per Match')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)


    avg_dfs = round(new_df['num_dfs_x'].mean(),0)

    card3 = dbc.Card([
            dbc.CardBody([
                html.H5(f'{avg_dfs}'),
                html.P('Avg # Double Faults Per Match')

            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)


    
    avg_bps = round(new_df['num_brkpts_saved_x'].mean(),0)

    card4 = dbc.Card([
            dbc.CardBody([
                html.H5(f'{avg_bps}'),
                html.P('Avg # Break Points Saved Per Match')
            ])
        ],
        style={
            'width': '100%',
            'text-align': 'center',
            'background-color': '#2E91E5',
            'color':'white',
            'fontWeight': 'bold',
            'fontSize':12},
        outline=True)

    return card1, card2, card3, card4

#app.run_server(host='0.0.0.0',port='8049')

if __name__=='__main__':
	app.run_server()