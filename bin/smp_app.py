import base64
import networkx as nx
import pandas as pd
from pathlib import Path
import PIL.Image
from st_aggrid import AgGrid
import streamlit as st
import tempfile
from typing import Any, BinaryIO, Dict, List, NamedTuple

from pysoc.sct.prefs import Profile, Ranking
from pysoc.sct.sct import SCF_COLLECTION
from pysoc.sct.smp import GaleShapleyAnimator, gale_shapley_weak, get_rank_signatures, make_reciprocal_suitee_profile, make_popular_suitee_profile

version = '0.1'

ROW_HEIGHT = 28
SQUASH = 0.9
THUMBNAIL_WIDTH = 100

RANKING_DESCRIPTIONS = {
    'Reciprocal/popular' : 'For each gift, rank people by strength of preference for that gift, then by gift popularity to break ties.',
    'Popular' : 'Rank people by gift popularity.',
    'Reciprocal' : 'For each gift, rank people by strength of preference for that gift.'
}

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


####################
# HELPER FUNCTIONS #
####################

def write_caption(s: str) -> None:
    st.markdown(f'<span style="font-size: 10pt; color: gray;">{s}</span>', unsafe_allow_html = True)

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
    if ('Brought' in df.columns):
        df['Brought'] = df['Brought'].fillna('')
    return df

def get_state(key: str, default: Any) -> Any:
    return getattr(st.session_state, key, default)

def set_state(key: str, val: Any) -> None:
    setattr(st.session_state, key, val)

def initialize_state() -> None:
    set_state('form_submitted', False)
    # set_state('show_animation', False)

def render_title() -> None:
    col1, _, col2 = st.columns([6, 1, 8])
    col1.title('Gift Matching Algorithm')
    logo_path = Path(__file__).parents[1] / 'app_logo' / 'logo.jpg'
    logo = PIL.Image.open(logo_path)
    size = logo.size
    height = 200
    width = int((size[1] / size[0]) * height)
    logo = logo.resize((height, width))
    col2.image(logo)
    # st.title('Gift Matching Algorithm')

def submit_form() -> None:
    set_state('form_submitted', True)
    set_state('show_animation', False)
    set_state('show_animation_link', False)

def render_ranking_form(n: int, should_rank_people: bool, have_csv: bool) -> pd.DataFrame:
    st.write('Enter user preference rankings.')
    with st.form('rankings form'):
        ranking_help = 'Provide each person\'s full ranking of gifts from favorite to least favorite, separated by commas (for gifts whose rankings are tied, you may separate by semicolons.'
        if should_rank_people:
            ranking_help += '<br>Additionally, please provide the gift brought by each person.'
        write_caption(ranking_help)
        df_template = st.session_state.table_data if have_csv else initialize_table(n, rank = should_rank_people)
        response = AgGrid(df_template, height = table_height(n), editable = True, fit_columns_on_grid_load = True)
        st.form_submit_button(on_click = submit_form)
    return response['data']

def parse_ranking(s: str) -> Ranking:
    return Ranking([[item.strip() for item in tok.split(';')] for tok in s.split(',')])

def get_winners(profile: Profile) -> pd.DataFrame:
    scfs, winners = [], []
    for scf in SCF_COLLECTION:
        scf_name = str(scf).replace('_', ' ').title()
        scfs.append(scf_name)
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

def gale_shapley_animator(suitors: List[str], suitees: List[str]) -> GaleShapleyAnimator:
    suitor_images = get_images(suitors, get_state('person_pics', []))
    suitee_images = get_images(suitees, get_state('gift_pics', []))
    n = len(suitors)
    width = min(11, 0.75 * n)
    height = max(2, 0.55 * width)
    return GaleShapleyAnimator(suitors, suitees, suitor_images = suitor_images, suitee_images = suitee_images, figsize = (width, height), thumbnail_width = 100)

# def animate_gale_shapley(suitors: List[str], suitees: List[str], anim_df: pd.DataFrame) -> FuncAnimation:
#     suitor_images = get_images(suitors, getattr(st.session_state, 'person_pics', []))
#     suitee_images = get_images(suitees, getattr(st.session_state, 'gift_pics', []))
#     n = len(suitors)
#     width = min(11, 0.75 * n)
#     height = max(1, 0.55 * width)
#     animator = GaleShapleyAnimator(suitors, suitees, suitor_images = suitor_images, suitee_images = suitee_images, figsize = (width, height), thumbnail_width = 100)
#     return animator.animate(anim_df)

class SMPOptions(NamedTuple):
    rank_suitors: str
    agg: str = 'borda'
    def rank_popularity(self) -> bool:
        return (self.rank_suitors != 'Reciprocal')
    def get_suitee_profile(self, suitor_profile: Profile, brought: List[str]) -> Profile:
        if (self.rank_suitors == 'Popular'):
            suitees_by_suitor = {}
            for (suitor, suitee) in zip(suitor_profile.names, brought):
                suitees_by_suitor[suitor] = suitee
            return make_popular_suitee_profile(suitor_profile, suitees_by_suitor, agg = self.agg)
        elif (self.rank_suitors == 'Reciprocal'):
            return make_reciprocal_suitee_profile(suitor_profile)
        else:  # reciprocal/popular
            reciprocal = SMPOptions('Reciprocal', self.agg).get_suitee_profile(suitor_profile, brought)
            popular = SMPOptions('Popular', self.agg).get_suitee_profile(suitor_profile, brought)
            rankings = [ranking1.refine(ranking2) for (ranking1, ranking2) in zip(reciprocal, popular)]
            return Profile(rankings, names = reciprocal.names)

class SMPData(NamedTuple):
    options: SMPOptions
    suitors: List[str]
    suitees: List[str]
    suitor_profile: Profile
    suitee_profile: Profile
    suitor_winners: pd.DataFrame
    suitee_winners: pd.DataFrame
    matching_graph: nx.Graph
    anim_actions: pd.DataFrame
    matches: pd.DataFrame
    @classmethod
    def from_form_data(cls, df: pd.DataFrame, options: SMPOptions) -> 'SMPData':
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
        brought = []
        if options.rank_popularity():  # validate list of brought items
            brought = list(df['Brought'])
            for suitee in suitees:
                if (suitee not in brought):
                    raise ValueError(f'Must indicate which person brought {suitee!r}')
            for (suitor, suitee) in zip(df['Person'], df['Brought']):
                if (suitee not in suitees):
                    raise ValueError(f'Invalid brought item for {suitor}: {suitee!r}')
        suitee_profile = options.get_suitee_profile(suitor_profile, brought)
        suitor_winners = get_winners(suitee_profile)
        suitee_winners = get_winners(suitor_profile)
        (graph, anim_actions) = gale_shapley_weak(suitor_profile, suitee_profile, random_tiebreak = True)
        matches = pd.DataFrame(graph.edges, columns = ['Person', 'Gift']).sort_values(by = 'Person', key = lambda s : [suitors.index(i) for i in s])
        return SMPData(options, suitors, suitees, suitor_profile, suitee_profile, suitor_winners, suitee_winners, graph, anim_actions, matches)
    @property
    def num_suitors(self) -> int:
        return len(self.suitors)
    def render_preferences(self) -> None:
        (col1, col2) = st.columns((1, 1))
        col1.markdown('__Gift Rankings__')
        suitor_profile = Profile([ranking.get_ranking() for ranking in self.suitor_profile], names = self.suitor_profile.names)
        suitor_df = suitor_profile.to_pandas()
        col1.dataframe(suitor_df, height = table_height(len(suitor_df)))
        col2.markdown('__Person Rankings__')
        suitee_profile = Profile([ranking.get_ranking() for ranking in self.suitee_profile], names = self.suitee_profile.names)
        suitee_df = suitee_profile.to_pandas()
        col2.dataframe(suitee_df, height = table_height(len(suitee_df)))
    def render_rcv(self) -> None:
        with st.expander('Ranked Choice Voting'):
            cols = st.columns((1, 1)) if self.options.rank_popularity() else [st]
            cols[0].markdown('__Gifts:__')
            cols[0].dataframe(self.suitee_winners, height = table_height(len(SCF_COLLECTION)))
            if self.options.rank_popularity():
                cols[1].markdown('__People:__')
                cols[1].dataframe(self.suitor_winners, height = table_height(len(SCF_COLLECTION)))
    def render_matching(self) -> None:
        with st.expander('Matching Results'):
            st.dataframe(self.matches, height = table_height(self.num_suitors))
            (suitor_sig, _) = get_rank_signatures(self.suitor_profile, self.suitee_profile, self.matching_graph)
            st.write(f'Rank Signature: {suitor_sig.signature}')
            happiness_score = round(suitor_sig.happiness_score)
            st.write(f'Happiness Score: {happiness_score}%')
    def render_show_animation(self) -> None:
        def clicked_show_animation():
            set_state('form_submitted', True)
            set_state('show_animation', True)
        st.button('Show Animation', on_click = clicked_show_animation)
        if get_state('show_animation', False):
            animator = gale_shapley_animator(self.suitors, self.suitees)
            animation = animator.animate(self.anim_actions)
            with st.spinner('Generating animation...'):
                anim_height = 100 + 100 * animator.figsize[1]
                st.components.v1.html(animation.to_jshtml(), height = anim_height)
    def render_download_animation(self) -> None:
        def clicked_download_animation():
            set_state('form_submitted', True)
            set_state('show_animation_link', True)
        st.button('Download Animation', on_click = clicked_download_animation)
        if get_state('show_animation_link', False):
            filename = 'animation.mp4'
            animator = gale_shapley_animator(self.suitors, self.suitees)
            animation = animator.animate(self.anim_actions, squash = SQUASH)
            with st.spinner('Generating animation...'):
                with tempfile.NamedTemporaryFile('wb+', suffix = '.mp4') as tf:
                    animation.save(tf.name, writer = 'ffmpeg', dpi = 400, fps = 2)
                    tf.flush()
                    tf.seek(0)
                    data = tf.read()
                    b64 = base64.b64encode(data).decode()
                    link = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{filename}</a>'
            st.markdown(link, unsafe_allow_html = True)
    def render_animation(self) -> None:
        self.render_show_animation()
        self.render_download_animation()


def main() -> None:
    render_title()
    with st.expander('Upload files (optional)'):
        csv_file = st.file_uploader('Upload CSV of rankings', type = ['csv'], on_change = initialize_state)
        st.session_state.person_pics = st.file_uploader('Upload pics of people', type = ['jpg', 'png'], accept_multiple_files = True, on_change = initialize_state)
        st.session_state.gift_pics = st.file_uploader('Upload pics of gifts', type = ['jpg', 'png'], accept_multiple_files = True, on_change = initialize_state)
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
    rank_people = st.selectbox('Ranking criterion?', ['Reciprocal/popular', 'Popular', 'Reciprocal'])
    rank_people_descr = RANKING_DESCRIPTIONS[rank_people]
    write_caption(RANKING_DESCRIPTIONS[rank_people])
    should_rank_people = rank_people != 'Reciprocal'
    table_data = render_ranking_form(n, should_rank_people, have_csv)
    options = SMPOptions(rank_people)
    st.download_button('Download CSV', table_data.to_csv(index = False), file_name = 'rankings.csv', mime = 'text/csv')
    if get_state('form_submitted', False):
        try:
            data = SMPData.from_form_data(table_data, options)
            data.render_preferences()
            data.render_rcv()
            data.render_matching()
            data.render_animation()
        except ValueError as e:
            st.error(f'Invalid input: {e}')

# run the app
main()