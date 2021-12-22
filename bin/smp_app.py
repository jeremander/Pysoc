import numpy as np
import pandas as pd
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
import streamlit as st
from typing import Any, Dict

from pysoc.sct.prefs import Profile, Ranking
from pysoc.sct.sct import SCF_COLLECTION
from pysoc.sct.smp import GaleShapleyAnimator, gale_shapley_weak, make_compliant_suitee_profile, make_popular_suitee_profile

version = '0.1'
ROW_HEIGHT = 28

st.set_page_config(page_title = 'Gift Matching', page_icon = '🎁', layout = 'wide')

# TEST_PATH = '/Users/jerm/Programming/festivus/2019-mugs/mug_suitors.csv'

# def load_test_data():
#     df = pd.read_csv(TEST_PATH, index_col = 0)
#     numeric_cols = [col for col in df.columns if col.isdigit()]
#     rankings = []
#     for (_, s) in df.iterrows():
#         ranking = []
#         for col in numeric_cols:
#             val = s.loc[col]
#             if isinstance(val, str):
#                 ranking.append(val)
#         rankings.append(','.join(ranking))
#     return pd.DataFrame({'Person' : df.index, 'Brought' : df.Brought, 'Ranked Items' : rankings})

# TEST_DF = load_test_data()
# TEST_DF = pd.DataFrame({'Person' : ['Alice', 'Bob', 'Charlie'], 'Brought' : ['A', 'B', 'C'], 'Ranked Items' : ['B,A,C','A,B,C','C,B,A']})

def table_height(num_rows: int) -> int:
    return 34 + ROW_HEIGHT * num_rows

def initialize_table(n: int, rank: bool = True) -> pd.DataFrame:
    people = [f'Person {i}' for i in range(1, n + 1)]
    empty_col = [''] * n
    d = {'Person' : people}
    if rank:
        d['Brought'] = empty_col
    d['Ranked Items'] = empty_col
    return pd.DataFrame(d)

def load_table_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    for col in ['Person', 'Ranked Items']:
        if (col not in cols):
            raise ValueError(f'CSV does not contain column {col!r}')
    return df

def parse_ranking(s: str) -> Ranking:
    return Ranking([[item.strip() for item in tok.split(';')] for tok in s.split(',')])

def get_winners(profile: Profile) -> pd.Series:
    scfs, winners = [], []
    for scf in SCF_COLLECTION:
        scfs.append(str(scf))
        winners.append(', '.join(scf(profile)))
    return pd.DataFrame({'SCF' : scfs, 'winners' : winners})

def animate_gale_shapley(suitors, suitees, anim_df):
    width = min(11, 0.75 * len(suitors))
    height = 0.55 * width
    animator = GaleShapleyAnimator(suitors, suitees, figsize = (width, height))
    anim = animator.animate(anim_df)
    st.components.v1.html(anim.to_jshtml(), height = 700)

def get_data(df: pd.DataFrame, rank_people: bool, agg: str = 'borda') -> Dict[str, Any]:
    suitors = list(df['Person'])
    num_suitors = len(suitors)
    rankings = []
    for (i, s) in enumerate(df['Ranked Items']):
        try:
            ranking = parse_ranking(s)
        except ValueError as e:
            suitor = suitors[i]
            raise ValueError(f'Ranking for {suitor}: {e}')
    rankings = [parse_ranking(s) for s in df['Ranked Items']]
    for (i, ranking) in enumerate(rankings):
        suitor = suitors[i]
        num_suitees = len(ranking.universe)
        if (num_suitees != num_suitors):
            raise ValueError(f'Incorrect number of ranked items for {suitor} (expected {num_suitors}, got {num_suitees})')
        if (ranking.universe != rankings[0].universe):
            raise ValueError(f'Ranked items for {suitor} inconsistent with prior items.')
    suitees = sorted(ranking.universe)
    suitor_profile = Profile(rankings, names = suitors)
    if rank_people:
        brought = set(suitees)
        for suitee in ranking.universe:
            if (suitee not in brought):
                raise ValueError(f'Must indicate which person brought {suitee!r}')
        suitees_by_suitor = dict(zip(df['Person'], df['Brought']))
        # TODO: set rank agg algorithm
        suitee_profile = make_popular_suitee_profile(suitor_profile, suitees_by_suitor, agg = agg)
    else:
        suitee_profile = make_compliant_suitee_profile(suitor_profile)
    (graph, anim_df) = gale_shapley_weak(suitor_profile, suitee_profile, random_tiebreak = True)
    matches = pd.DataFrame(graph.edges, columns = ['Person', 'Item']).sort_values(by = 'Person', key = lambda s : [suitors.index(i) for i in s])
    suitor_winners = get_winners(suitee_profile)
    suitee_winners = get_winners(suitor_profile)
    return {'suitors' : suitors, 'suitees' : suitees, 'suitor_profile' : suitor_profile, 'suitee_profile' : suitee_profile, 'matches' : matches, 'anim_df' : anim_df, 'suitor_winners' : suitor_winners, 'suitee_winners' : suitee_winners}

st.title('Gift Matching Algorithm')

csv_path = st.file_uploader('Upload CSV (optional)')

if (csv_path is not None):
    try:
        st.session_state.table_data = load_table_from_csv(csv_path)
    except ValueError as e:
        st.error(e)

have_csv = 'table_data' in st.session_state
if have_csv:
    n = len(st.session_state.table_data)
else:
    n = int(st.number_input('How many people?', min_value = 1, value = 1, format = '%d'))

rank_people = st.radio('Rank people by gift popularity?', ['Yes', 'No']) == 'Yes'

st.write('Enter user preference rankings.')

def submit_form() -> None:
    st.session_state.form_submitted = True

with st.form('rankings form') as f:
    df_template = st.session_state.table_data if have_csv else initialize_table(n, rank = rank_people)
    response = AgGrid(df_template, height = table_height(n), editable = True, fit_columns_on_grid_load = True)
    st.form_submit_button(on_click = submit_form)

table_data = response['data']
st.download_button('Download CSV', table_data.to_csv(index = False), file_name = 'rankings.csv', mime = 'text/csv')

if getattr(st.session_state, 'form_submitted', False):
    try:
        data = get_data(table_data, rank_people)
        st.markdown('__Best items:__')
        st.dataframe(data['suitee_winners'], height = table_height(len(SCF_COLLECTION)))
        if rank_people:
            st.markdown('__Best people:__')
            st.dataframe(data['suitor_winners'], height = table_height(len(SCF_COLLECTION)))
        st.markdown('__Matching:__')
        st.dataframe(data['matches'], height = table_height(n))
        with st.spinner('Generating animation...'):
            animate_gale_shapley(data['suitors'], data['suitees'], data['anim_df'])
    except ValueError as e:
        st.error(f'Invalid input: {e}')
