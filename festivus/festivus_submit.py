from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

from pysoc.sct.prefs import Ranking


# TODO: move these to config file
MIN_CHOICES = 5
SELF_RANKED_LAST = True

st.set_page_config(page_title = 'Festivus!', page_icon = '🎁')

conn = st.connection('gsheets', type = GSheetsConnection)


def validate_input(name: str, brought: str, description: str, ranking: str) -> str:
    """Validates user input fields.
    Returns the canonical ranking string."""
    if (not name):
        raise ValueError('Please input your name.')
    if (not brought):
        raise ValueError("Please provide your gift's letter code.")
    if (not brought.isalpha()) or (len(brought) > 1):
        raise ValueError('Gift code must be a single letter.')
    if (not description):
        raise ValueError('Please provide a description.')
    for c in ranking:
        if not (c.isalpha() or c.isspace() or (c == '/')):
            raise ValueError(f'Invalid character {c!r} in ranking.')
    items = []
    toks = ranking.strip().split()
    for tok in toks:
        tier = tok.split('/')
        for item in tier:
            if (not item.isalpha()) or (len(item) > 1):
                raise ValueError(f'Error in ranking: {item!r} is not a letter.')
        items.append(tier)
    ranking_obj = Ranking(items)
    if (len(ranking_obj) == 0):
        raise ValueError('Please provide your ranking.')
    if (len(ranking_obj.universe) < MIN_CHOICES):
        raise ValueError(f'Must rank at least {MIN_CHOICES} items.')
    if SELF_RANKED_LAST and (brought in ranking_obj.universe):
        raise ValueError(f'You may not include your own gift ({brought}) in the ranking.')
    return str(ranking_obj).replace(' ', '')


def main() -> None:
    year = datetime.now().year
    st.header(f'Festivus {year} Gift Ranking')
    name = st.text_input('What is your name?', key = 'name')
    brought = st.text_input('What is the letter code of the gift you brought?')
    description = st.text_input('Give a brief description of your gift.')
    st.write('')
    instruction_lines = [
        'Rank the gifts from favorite to least favorite, separated by spaces.',
        f'You must rank at least {MIN_CHOICES} gifts.',
        'For gifts whose rankings are tied, you may separate them by slashes.',
        'Example: B C A/E D'
    ]
    if SELF_RANKED_LAST:
        instruction_lines.insert(2, 'Per Festivus tradition, you may not include your own gift.')
    st.markdown('\n'.join(f'- {line}' for line in instruction_lines))
    # st.caption('<br>'.join(caption_lines), unsafe_allow_html = True)
    ranking = st.text_input('Gift ranking:')
    st.write('')
    if st.button('Submit', type = 'primary'):
        try:
            ranking = validate_input(name, brought, description, ranking)
        except ValueError as e:
            st.error(str(e))
            return
        df = conn.read(ttl = 0).dropna(how = 'all')
        if (df['Person'] == name).any():
            st.error(f'Choices already submitted for {name!r}. Please reset data in Google spreadsheet.')
            return
        row = {'Person': name, 'Brought': brought, 'Description': description, 'Ranked Gifts': ranking}
        df = pd.concat([df, pd.DataFrame.from_records([row])])
        with st.spinner('Submitting data...'):
            # st.dataframe(df)
            conn.update(data = df)
        st.write('Your choices have been submitted! 🎉')

main()
