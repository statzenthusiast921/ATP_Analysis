#Import packages
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import os
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

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

#Player --> Surface Dictionary
player_surface_df = atp_df[['player_name','surface']].drop_duplicates()
player_surface_dict = player_surface_df.groupby('player_name')['surface'].agg(list).to_dict()


#----- Model Code
from xgboost import XGBClassifier

#Choose features and response
y = atp_df['outcome']
X = atp_df[['num_aces','num_dfs','serve1_in_perc','player_age','surface','num_brkpts_saved','num_brkpts_faced']]

#One Hot Encode surface
one_hot = pd.get_dummies(X['surface'])
X = X.drop('surface',axis=1)
X = X.join(one_hot)


scale = StandardScaler()
scaledX = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaledX, y, test_size=0.30, random_state=42)


xgb_class = XGBClassifier()
xgb_class.fit(X_train, y_train, verbose = False, early_stopping_rounds=15,eval_set=[(X_test,y_test)])
y_pred = xgb_class.predict(scaledX)

atp_df['pred_wins'] = y_pred


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
                        html.Div([
                            dbc.Button("Click Here for Instructions", id="open1",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                    dbc.ModalHeader("Instructions"),
                                    dbc.ModalBody(
                                        children=[
                                            html.P('Below is a player-specific table with details about each ATP match from 1991 to 2022.'),
                                            html.P('You can update the table by selecting a player, surface, and timeframe.'),
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        dbc.Button("Close", id="close1", className="ml-auto")
                                    ),
                            ],id="modal1",size="md",scrollable=True),
                        ],className="d-grid gap-2")
                    ],width=12)
                ]),
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
                    html.Div([
                        dbc.Button("Click Here for Instructions", id="open2",color='secondary',style={"fontSize":18}),
                        dbc.Modal([
                            dbc.ModalHeader("Instructions"),
                            dbc.ModalBody(
                                children=[
                                    html.P('Below is a chart showcasing how the selected player has performed over time utilizing several statistics.'),
                                    html.P('You can update the chart by selecting a player and one of the 8 available statistics.  Data is aggregated at the quarterly level.'),
                                ]
                            ),
                            dbc.ModalFooter(
                                dbc.Button("Close", id="close2", className="ml-auto")
                            ),
                        ],id="modal2",size="md",scrollable=True),
                    ],className="d-grid gap-2")
                ],width=12)
            ]),
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
                        value='Rafael Nadal'
                    )
                ],width=6),  
                dbc.Col([
                 #----- Statistic filter
                    dcc.Dropdown(
                        id='dropdown3',
                        style={'color':'black'},
                        #multi = True,
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
                        html.Div([
                            dbc.Button("Click Here for Instructions", id="open3",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                dbc.ModalHeader("Instructions"),
                                dbc.ModalBody(
                                    children=[
                                        html.P('Below is a chart showcasing the cumulative wins earned for the selected player against the selected opponent.  Average match statistics are also presented.'),
                                        html.P('You can update the chart and statistics by selecting a player and then select any of his opponents.'),
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="close3", className="ml-auto")
                                ),
                            ],id="modal3",size="md",scrollable=True),
                        ],className="d-grid gap-2")
                    ],width=12)
                ]),
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
                    #----- Opponent filter
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
                        dbc.Card(id = 'card0')
                    ])
                ]),
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
                dbc.Row([
                    dcc.Graph(id = 'cumulative_wins')
                ])
             

            ]
        ),
        dcc.Tab(label='Predict Winners',value='tab-5',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Button("Click Here for Instructions", id="open4",color='secondary',style={"fontSize":18}),
                            dbc.Modal([
                                dbc.ModalHeader("Instructions"),
                                dbc.ModalBody(
                                    children=[
                                        html.P('Below is a chart and table showcasing the results of fitting an XGBoost model to match-level statistics in order to predict the outcome of any ATP match.  In order to update the page, select a player and select a combination of surfaces.'),
                                        html.P('The model used the number of aces, double faults, 1st serve in %, age of the player, surface, and the number of break points saved and faced to predict the outcome of the match.'),
                                        html.P('The chart compares the actual outcome for the selected player (blue) and the predicted outcome (green).  The performance of the model can be analyzed using the 4 statistics above the chart and the confusion matrix to the right of the chart.'),
                                        html.P('Accuracy measures the # of correct predictions (wins and losses) out of all predictions.  Precision measures how many predicted wins were correct out of all predicted wins (wins that were correct and incorrect).  Recall measures the number of correct predictions of wins out of all the predictions that should be wins (correct wins and wins that were predicted as losses). The F1 score measures the balance between precision and recall.')
                                    ]
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="close4", className="ml-auto")
                                ),
                            ],id="modal4",size="md",scrollable=True),
                        ],className="d-grid gap-2")
                    ],width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose a player:')
                    ], width = 6),
                    dbc.Col([
                        dbc.Label('Choose a surface:')
                    ], width = 6),
                ]),
                dbc.Row([
                    dbc.Col([
                    #----- Player filter
                        dcc.Dropdown(
                            id='dropdown6',
                            options=[{'label': i, 'value': i} for i in player_choices],
                            value = 'Roger Federer'
                        )
                    ],width=6),
                    dbc.Col([
                    #----- Surface filter
                        dcc.Dropdown(
                            id='dropdown7',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in surface_choices],
                            value = surface_choices[0:4],
                            multi = True
                        )
                    ],width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id="card5")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card6")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card7")
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id="card8")
                    ],width=3)
                ],className="g-0"),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id = 'predicted_wins')
                    ], width = 8),
                    dbc.Col([
                        dcc.Graph(id = 'confusion_matrix')
                    ],width = 4)
                ])
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

    
#----- Tab #2: Master matches table filterable by player, surface, and
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

#----- Tab 3: Individual Stats filterable by player and specific statistic

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


    stats_df = filtered.groupby(['tourney_date','surface']).agg({
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

    full_days_df['year'] = full_days_df['tourney_date'].astype(str).str[0:4]
    full_days_df['quarter'] = full_days_df['tourney_date'].dt.quarter
    full_days_df['yq'] = full_days_df['year'].astype(str) + "Q" + full_days_df['quarter'].astype(str)
    full_days_df['quarter_date'] = pd.PeriodIndex(
        full_days_df['tourney_date'], freq='Q'
    ).to_timestamp()



    line_chart_df = full_days_df.groupby(['quarter_date','surface']).agg({
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
                color = 'surface',
                x="quarter_date", 
                y="game_win_perc", 
                markers=True,
                template = 'plotly_dark',
                labels={"quarter_date": "Month-Year (Q)",
                        "game_win_perc": "% Games Won"
                }
        )

        line_chart.update_layout(
            title_x=0.5,
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.075,
                xanchor="center",
                x=0.5
            )
        )
        return line_chart

    #----- Stat #2: 1st Serve in %
    elif statistic_choices[1] in dd3:

        line_chart = px.line(
                line_chart_df, 
                color = 'surface',
                x="quarter_date", 
                y="serve1_in_perc", 
                markers=True,
                template = 'plotly_dark',
                labels={"quarter_date": "Month-Year (Q)",
                        "serve1_in_perc": "1st Serve in %"
                }
        )
        line_chart.update_layout(
            title_x=0.5,
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.075,
                xanchor="center",
                x=0.5
            )
        )

        return line_chart

    #----- Stat #3: 1st Serve Win %
    elif statistic_choices[2] in dd3:
    
        line_chart = px.line(
                line_chart_df, 
                color = 'surface',
                x="quarter_date", 
                y="serve1_win_perc", 
                markers=True,
                template = 'plotly_dark',
                labels={"quarter_date": "Month-Year (Q)",
                        "serve1_win_perc": "1st Serve Win %"
                }
        )
        line_chart.update_layout(
            title_x=0.5,
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.075,
                xanchor="center",
                x=0.5
            )
        )

        return line_chart

    #----- Stat #4: 2nd Serve Win %
    elif statistic_choices[3] in dd3:
        
        line_chart = px.line(
                line_chart_df, 
                color = 'surface',
                x="quarter_date", 
                y="serve2_win_perc", 
                markers=True,
                template = 'plotly_dark',
                labels={"quarter_date": "Month-Year (Q)",
                        "serve2_win_perc": "2nd Serve Win %"
                }
        )
        line_chart.update_layout(
            title_x=0.5,
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.075,
                xanchor="center",
                x=0.5
            )
        )

        return line_chart

    #----- Stat #5: Aces
    elif statistic_choices[4] in dd3:
            
        line_chart = px.line(
                line_chart_df, 
                color = 'surface',
                x="quarter_date", 
                y="num_aces", 
                markers=True,
                template = 'plotly_dark',
                labels={"quarter_date": "Month-Year (Q)",
                        "num_aces": "# Aces"
                }
        )
        line_chart.update_layout(
            title_x=0.5,
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.075,
                xanchor="center",
                x=0.5
            )
        )

        return line_chart

    #----- Stat #6: Break Points Faced
    elif statistic_choices[5] in dd3:
            
        line_chart = px.line(
                line_chart_df, 
                color = 'surface',
                x="quarter_date", 
                y="num_brkpts_faced", 
                markers=True,
                template = 'plotly_dark',
                labels={"quarter_date": "Month-Year (Q)",
                        "num_brkpts_faced": "# Break Points Faced"
                }
            )
        line_chart.update_layout(
            title_x=0.5,
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.075,
                xanchor="center",
                x=0.5
            )
        )

        return line_chart

    #----- Stat #7: Break Points Saved
    elif statistic_choices[6] in dd3:
            
        line_chart = px.line(
                line_chart_df, 
                color = 'surface',
                x="quarter_date", 
                y="num_brkpts_saved", 
                markers=True,
                template = 'plotly_dark',
                labels={"quarter_date": "Month-Year (Q)",
                        "num_brkpts_saved": "# Break Points Saved"
                }
        )

        line_chart.update_layout(
            title_x=0.5,
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.075,
                xanchor="center",
                x=0.5
            )
        )

        return line_chart



    #----- Stat #8: # Double Faults
    elif statistic_choices[7] in dd3:
            
        line_chart = px.line(
                line_chart_df, 
                color = 'surface',
                x="quarter_date", 
                y="num_dfs", 
                markers=True,
                template = 'plotly_dark',
                labels={"quarter_date": "Month-Year (Q)",
                        "num_dfs": "# Double Faults"
                }
        )

        line_chart.update_layout(
            title_x=0.5,
            legend_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.075,
                xanchor="center",
                x=0.5
            )
        )


        return line_chart


#--- Set up a dependent dropdown menu for head to head tab (tab 4) - player vs. opponent
@app.callback(
    Output('dropdown5', 'options'),#-----Filters the opponent options
    Output('dropdown5', 'value'),
    Input('dropdown4', 'value') #----- Select the player
)
def set_character_options(selected_player):

    if selected_player == "Roger Federer":
        return [{'label': i, 'value': i} for i in player_opponents_dict[selected_player]], player_opponents_dict[selected_player][196]
    else:

        return [{'label': i, 'value': i} for i in player_opponents_dict[selected_player]], player_opponents_dict[selected_player][0]


#----- Callback to update all the head-to-head statistics on tab 4
@app.callback(
    Output('card0', 'children'),
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

    win_df = new_df[new_df['outcome_x']==1]
    loss_df = new_df[new_df['outcome_x']==0]

    wins = win_df.shape[0]
    losses = loss_df.shape[0]

    card0 = dbc.Card([
            dbc.CardBody([
                html.H5(f'How did {dd4} fare against {dd5}?'),
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

    return card0, card1, card2, card3, card4


#Callback on tab 4 to set up step chart for cumulative wins (player vs. opponent)
@app.callback(
    Output('cumulative_wins','figure'),
    Input('dropdown4','value'),
    Input('dropdown5','value')
)
def cumulative_wins(dd4, dd5):

    player1_df = atp_df[atp_df['player_name']==dd4]
    player2_df = atp_df[atp_df['player_name']==dd5]

    #player1_df = atp_df[atp_df['player_name']=="Rafael Nadal"]
    #player2_df = atp_df[atp_df['player_name']=="Roger Federer"]

    new_df = pd.merge(
        player1_df,
        player2_df,
        how = 'inner',
        on = ['tourney_id','match_num']
    )

    cum_win_df  = new_df[[
        'tourney_id','tourney_name_x','surface_x', 'tourney_date_x',
        'player_name_x','outcome_x','player_name_y','outcome_y'
    ]]
    cum_win_df = cum_win_df.sort_values(['tourney_date_x'], ascending=True)

    cum_win_df['cum_wins_x'] = cum_win_df['outcome_x'].cumsum()
    cum_win_df['cum_wins_y'] = cum_win_df['outcome_y'].cumsum()
    cum_win_df['Match #'] = range(len(cum_win_df))


    df1 = cum_win_df[[
        'tourney_id','tourney_name_x','surface_x',
        'tourney_date_x','player_name_x','cum_wins_x',
        'Match #'
    ]]
    df1 = df1.rename(columns={
        'player_name_x': "player_name",
        'cum_wins_x':'cum_wins'
    })

    df2 = cum_win_df[[
        'tourney_id','tourney_name_x','surface_x',
        'tourney_date_x','player_name_y','cum_wins_y',
        'Match #'
    ]]
    df2 = df2.rename(columns={
        'player_name_y': "player_name",
        'cum_wins_y':'cum_wins'
    })

    df_stacked = pd.concat([df1, df2])
    df_stacked['Match #'] = df_stacked['Match #'] + 1 
    df_stacked['tourney_date_x'] = pd.to_datetime(df_stacked['tourney_date_x'], format='%Y%m%d').dt.date
    
    line_chart = px.line(
        df_stacked, 
        x = "Match #", 
        color = 'player_name',
        y = "cum_wins", 
        markers=True,
        template = 'plotly_dark',
        hover_data = {
            "player_name":True,
            "tourney_name_x":True,
            "tourney_date_x":True,
            "surface_x":True,
            "cum_wins":True,
            "tourney_id":False,
            #"# Match":False
        },
        labels={
            "tourney_id": "Tourney-ID",
            "tourney_name_x": "Tourney Name",
            "surface_x": "Surface",
            "tourney_date_x":"Tourney Date",
            "player_name": "Player Name",
            "cum_wins":"Cumulative Wins"
        }
    )

    line_chart.update_layout(
        title_text=f"Cumulative Games Won ({dd4} vs. {dd5})", 
        title_x=0.5,
        legend_title=None,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.075,
            xanchor="center",
            x=0.5
        )
    )

    line_chart.update_xaxes(type='category')

    return line_chart

#----- Callback for everything on tab 5 - XGBoost model results

@app.callback(
    Output('predicted_wins','figure'),
    Output('confusion_matrix','figure'),
    Output('card5','children'),
    Output('card6','children'),
    Output('card7','children'),
    Output('card8','children'),
    Input('dropdown6','value'),
    Input('dropdown7','value')
)
def pred_cumulative_wins(dd6, dd7):

    player_df = atp_df[atp_df['player_name']==dd6]
    surface_player_df = player_df[player_df['surface'].isin(dd7)]

    #player_df = atp_df[atp_df['player_name']=="Rafael Nadal"]
   
    pred_cum_win_df  = surface_player_df[[
        'tourney_id','tourney_name','surface', 'tourney_date',
        'player_name','outcome','pred_wins'
    ]]
    pred_cum_win_df = pred_cum_win_df.sort_values(['tourney_date'], ascending=True)

    pred_cum_win_df['Match #'] = range(len(pred_cum_win_df))


    pred_cum_win_df['Match #'] = pred_cum_win_df['Match #'] + 1 
    pred_cum_win_df['tourney_date'] = pd.to_datetime(pred_cum_win_df['tourney_date'], format='%Y%m%d').dt.date
    
    actuals = pred_cum_win_df.loc[:, pred_cum_win_df.columns != 'pred_wins']
    actuals['type'] = "Actual"
    predictions = pred_cum_win_df.loc[:, pred_cum_win_df.columns != 'outcome']
    predictions['type'] = "Prediction"


    actuals = actuals.rename(
        columns={"outcome": "Outcome"}
    )

    predictions = predictions.rename(
        columns={"pred_wins": "Outcome"}
    )


    actuals['cum_wins'] = actuals['Outcome'].cumsum()
    predictions['cum_wins'] = predictions['Outcome'].cumsum()

    df_stacked = pd.concat([actuals, predictions])


    line_chart = px.line(
        df_stacked, 
        x = "Match #", 
        color = 'type',
        y = "cum_wins", 
        markers=True,
        template = 'plotly_dark',
        hover_data = {
            "tourney_name":True,
            "tourney_date":True,
            "surface":True,
            "cum_wins":True,
            "tourney_id":False,
            #"# Match":False
        },
        labels={
            "tourney_id": "Tourney-ID",
            "tourney_name": "Tourney Name",
            "surface": "Surface",
            "tourney_date":"Tourney Date",
            "cum_wins":"Cumulative Wins"
        }
    )

    line_chart.update_layout(
        title_text=f"{dd6} Predicted Wins vs. Actual Wins", 
        title_x=0.5,
        legend_title=None,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.075,
            xanchor="center",
            x=0.5
        )
    )

    line_chart.update_xaxes(type='category')
    #line_chart['data'][1]['line']['dash'] = 'dash'
    #line_chart['data'][0]['line']['color']='#2DFE54'

    line_chart['data'][1]['line']['color']='#2DFE54'

    #----- Measuring how well the predictions are doing
    cm_df = pred_cum_win_df[['outcome','pred_wins']]

    conditions = [
        ((cm_df['outcome'] == 1) & (cm_df['pred_wins'] == 1)),
        ((cm_df['outcome'] == 0) & (cm_df['pred_wins'] == 0)),
        ((cm_df['outcome'] == 1) & (cm_df['pred_wins'] == 0)),
        ((cm_df['outcome'] == 0) & (cm_df['pred_wins'] == 1))
    ]

    values = [
        'Actual Win, Pred Win', 
        'Actual Win, Pred Loss', 
        'Actual Loss, Pred Loss', 
        'Actual Loss, Pred Win'
    ]

    cm_df['cm'] = np.select(conditions, values)

    matrix = np.array([
        [cm_df['cm'].value_counts()[0],cm_df['cm'].value_counts()[1]],
        [cm_df['cm'].value_counts()[3],cm_df['cm'].value_counts()[2]]
    ])

    Index= ['Actual Win', 'Actual Loss']
    Cols = ['Predicted Win', 'Predicted Loss']
    heat_map_df = pd.DataFrame(matrix, index=Index, columns=Cols)

    heat_map = px.imshow(
        heat_map_df,
        template = 'plotly_dark',
        text_auto=True

    )
    heat_map.update_xaxes(side="top")

    true_pos = heat_map_df['Predicted Win'][0] 
    false_pos = heat_map_df['Predicted Win'][1] 
    false_neg = heat_map_df['Predicted Loss'][0] 
    true_neg = heat_map_df['Predicted Loss'][1] 

    accuracy = round(((true_pos + true_neg) / (true_pos + false_pos + false_neg + true_neg) ) *100,1)
    precision = round(((true_pos) / (true_pos + false_pos) ) *100,1)
    recall = round(((true_pos) / (true_pos + false_neg) ) *100,1)
    f1_score = round(2*((precision*recall)/(precision+recall)),1)

    card5 = dbc.Card([
            dbc.CardBody([
                html.H5(f'{accuracy}%'),
                html.P('Accuracy')
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

    card6 = dbc.Card([
            dbc.CardBody([
                html.H5(f'{precision}%'),
                html.P('Precision')
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

    card7 = dbc.Card([
            dbc.CardBody([
                html.H5(f'{recall}%'),
                html.P('Recall')
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

    card8 = dbc.Card([
            dbc.CardBody([
                html.H5(f'{f1_score}%'),
                html.P('F1 Score')
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

    return line_chart, heat_map, card5, card6, card7, card8


#----------Configure reactivity for Button #1 (Instructions) --> Tab #2----------#
@app.callback(
    Output("modal1", "is_open"),
    Input("open1", "n_clicks"), 
    Input("close1", "n_clicks"),
    State("modal1", "is_open")
)

def toggle_modal1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open    


#----------Configure reactivity for Button #2 (Instructions) --> Tab #3----------#
@app.callback(
    Output("modal2", "is_open"),
    Input("open2", "n_clicks"), 
    Input("close2", "n_clicks"),
    State("modal2", "is_open")
)

def toggle_modal2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open  

#----------Configure reactivity for Button #3 (Instructions) --> Tab #4----------#
@app.callback(
    Output("modal3", "is_open"),
    Input("open3", "n_clicks"), 
    Input("close3", "n_clicks"),
    State("modal3", "is_open")
)

def toggle_modal3(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open  

#----------Configure reactivity for Button #4 (Instructions) --> Tab #5----------#
@app.callback(
    Output("modal4", "is_open"),
    Input("open4", "n_clicks"), 
    Input("close4", "n_clicks"),
    State("modal4", "is_open")
)

def toggle_modal3(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open  


#app.run_server(host='0.0.0.0',port='8049')

if __name__=='__main__':
	app.run_server()