from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

from pysoc.sct.prefs import Ranking


MIN_CHOICES = 5

conn = st.connection('gsheets', type = GSheetsConnection)

def main():
    year = datetime.now().year
    st.header(f'Festivus {year} Gift Ranking')
    name = st.text_input('What is your name?', key = 'name')
    brought = st.text_input('What is the letter code of the gift you brought?')
    description = st.text_input('Give a brief description of your gift.')
    ranking = st.text_input('Gift ranking:')
    st.caption('Rank the gifts from favorite to least favorite, separated by spaces. For gifts whose rankings are tied, you may separate them by slashes. Example: B C A/E D')
    if st.button('Submit', type = 'primary'):
        if (not name):
            st.error('Please input your name.')
            return
        if (not brought):
            st.error("Please provide your gift's letter code.")
            return
        if (not brought.isalpha()) or (len(brought) > 1):
            st.error('Gift code must be a single letter.')
            return
        if (not description):
            st.error('Please provide a description.')
            return
        for c in ranking:
            if not (c.isalpha() or c.isspace() or (c == '/')):
                st.error(f'Invalid character {c!r} in ranking.')
                return
        items = []
        toks = ranking.strip().split()
        for tok in toks:
            tier = tok.split('/')
            for item in tier:
                if (not item.isalpha()) or (len(item) > 1):
                    st.error(f'Error in ranking: {item!r} is not a letter')
                    return
            items.append(tier)
        try:
            ranking = Ranking(items)
        except ValueError as e:
            st.error(str(e))
            return
        if (len(ranking) == 0):
            st.error('Please provide your ranking.')
            return
        df = conn.read(ttl = 0).dropna(how = 'all')
        if (df['Person'] == name).any():
            st.error(f'Choices already submitted for {name!r}. Please reset data in Google spreadsheet.')
            return
        row = {'Person': name, 'Brought': brought, 'Description': description, 'Ranked Gifts': str(ranking)}
        df = pd.concat([df, pd.DataFrame.from_records([row])])
        with st.spinner('Submitting data...'):
            # st.dataframe(df)
            conn.update(data = df)
        st.write('Your choices have been submitted! 🎉')

main()
