import pandas as pd
from pathlib import Path
from st_aggrid import AgGrid
import streamlit as st
from typing import Any, BinaryIO, Dict, List

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
#     return pd.DataFrame({'Person' : df.index, 'Brought' : df.Brought, 'Ranked Gifts' : rankings})

# TEST_DF = load_test_data()
TEST_DF = pd.DataFrame({'Person' : ['Alice', 'Bob', 'Charlie'], 'Brought' : ['A', 'B', 'C'], 'Ranked Gifts' : ['B,A,C','A,B,C','C,B,A']})

def normalize(s: str) -> str:
    return s.lower().replace('_', ' ')

def normalize_path(path: str) -> str:
    return normalize(Path(path).stem)

def table_height(num_rows: int) -> int:
    return 34 + ROW_HEIGHT * num_rows

def initialize_table(n: int, rank: bool = True) -> pd.DataFrame:
    people = [f'Person {i}' for i in range(1, n + 1)]
    empty_col = [''] * n
    d = {'Person' : people}
    if rank:
        d['Brought'] = empty_col
    d['Ranked Gifts'] = empty_col
    # return TEST_DF
    return pd.DataFrame(d)

def load_table_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)
    for col in ['Person', 'Ranked Gifts']:
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
    return pd.DataFrame({'scheme' : scfs, 'winners' : winners})

def get_images(names: List[str], files: List[st.uploaded_file_manager.UploadedFile]) -> Dict[str, BinaryIO]:
    d = {}
    names_by_key = {normalize(name) : name for name in names}
    for f in files:
        key = normalize_path(f.name)
        if (key in names_by_key):
            d[names_by_key[key]] = f
    if (bool(d) and (len(d) != len(names))):  # warn about missing images
        missing_names = ', '.join([name for name in names if (name not in d)])
        st.warning(f'Warning: Missing image files for {missing_names}')
    return d

def animate_gale_shapley(suitors: List[str], suitees: List[str], anim_df: pd.DataFrame) -> None:
    suitor_images = get_images(suitors, getattr(st.session_state, 'person_pics', []))
    suitee_images = get_images(suitees, getattr(st.session_state, 'gift_pics', []))
    n = len(suitors)
    width = min(11, 0.75 * n)
    height = max(1, 0.55 * width)
    animator = GaleShapleyAnimator(suitors, suitees, suitor_images = suitor_images, suitee_images = suitee_images, figsize = (width, height), thumbnail_width = 100)
    anim = animator.animate(anim_df)
    print(height)
    # 14, 6.05, 700
    st.components.v1.html(anim.to_jshtml(), height = 100 + 100 * height)

def get_data(df: pd.DataFrame, rank_people: bool, agg: str = 'borda') -> Dict[str, Any]:
    suitors = list(df['Person'])
    num_suitors = len(suitors)
    rankings = []
    for (i, s) in enumerate(df['Ranked Gifts']):
        suitor = suitors[i]
        try:
            if (not s):
                raise ValueError('No gifts listed.')
            ranking = parse_ranking(s)
        except ValueError as e:
            raise ValueError(f'Ranking for {suitor}: {e}')
    rankings = [parse_ranking(s) for s in df['Ranked Gifts']]
    for (i, ranking) in enumerate(rankings):
        suitor = suitors[i]
        num_suitees = len(ranking.universe)
        if (num_suitees != num_suitors):
            raise ValueError(f'Incorrect number of ranked gifts for {suitor} (expected {num_suitors}, got {num_suitees})')
        if (ranking.universe != rankings[0].universe):
            raise ValueError(f'Ranked gifts for {suitor} inconsistent with prior rows.')
    suitees = sorted(ranking.universe)
    suitor_profile = Profile(rankings, names = suitors)
    if rank_people:
        brought = set(suitees)
        for suitee in ranking.universe:
            if (suitee not in brought):
                raise ValueError(f'Must indicate which person brought {suitee!r}')
        suitees_by_suitor = {}
        for (suitor, suitee) in zip(df['Person'], df['Brought']):
            if (suitee not in suitees):
                raise ValueError(f'Invalid brought item for {suitor}: {suitee!r}')
            suitees_by_suitor[suitor] = suitee
        # TODO: set rank agg algorithm
        suitee_profile = make_popular_suitee_profile(suitor_profile, suitees_by_suitor, agg = agg)
    else:
        suitee_profile = make_compliant_suitee_profile(suitor_profile)
    (graph, anim_df) = gale_shapley_weak(suitor_profile, suitee_profile, random_tiebreak = True)
    matches = pd.DataFrame(graph.edges, columns = ['Person', 'Gift']).sort_values(by = 'Person', key = lambda s : [suitors.index(i) for i in s])
    suitor_winners = get_winners(suitee_profile)
    suitee_winners = get_winners(suitor_profile)
    return {'suitors' : suitors, 'suitees' : suitees, 'suitor_profile' : suitor_profile, 'suitee_profile' : suitee_profile, 'matches' : matches, 'anim_df' : anim_df, 'suitor_winners' : suitor_winners, 'suitee_winners' : suitee_winners}

st.title('Gift Matching Algorithm')

with st.expander('Upload files (optional)', expanded = False):
    csv_file = st.file_uploader('Upload CSV of rankings', type = ['csv'])
    st.session_state.person_pics = st.file_uploader('Upload pics of people', type = ['jpg', 'png'], accept_multiple_files = True)
    st.session_state.gift_pics = st.file_uploader('Upload pics of gifts', type = ['jpg', 'png'], accept_multiple_files = True)
    if (csv_file is not None):
        try:
            st.session_state.table_data = load_table_from_csv(csv_file)
        except ValueError as e:
            st.error(e)

have_csv = 'table_data' in st.session_state
if have_csv:
    n = len(st.session_state.table_data)
else:
    n = int(st.number_input('How many people?', min_value = 1, value = 1, format = '%d'))
    # n = len(TEST_DF)

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
        st.subheader('Ranked Choice Voting')
        cols = st.columns((1, 1)) if rank_people else [st]
        cols[0].markdown('__Gifts:__')
        cols[0].dataframe(data['suitee_winners'], height = table_height(len(SCF_COLLECTION)))
        if rank_people:
            cols[1].markdown('__People:__')
            cols[1].dataframe(data['suitor_winners'], height = table_height(len(SCF_COLLECTION)))
        st.markdown('__Matching:__')
        st.dataframe(data['matches'], height = table_height(n))
        with st.spinner('Generating animation...'):
            animate_gale_shapley(data['suitors'], data['suitees'], data['anim_df'])
    except ValueError as e:
        st.error(f'Invalid input: {e}')
