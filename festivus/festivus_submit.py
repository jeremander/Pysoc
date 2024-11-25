from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

from pysoc.img_util import img_from_bytes, img_to_base64, make_thumbnail
from pysoc.sct.prefs import Ranking


# TODO: move these to config file
MIN_CHOICES = 5
SELF_RANKED_LAST = True
IMG_TYPES = ['png', 'jpg', 'heic', 'heif']

st.set_page_config(page_title='Festivus!', page_icon='🎁')

conn = st.connection('gsheets', type=GSheetsConnection)


def normalize_name(name: str) -> str:
    """Normalizes a name."""
    return name.strip()

def validate_input(df: pd.DataFrame, name: str, brought: str, description: str, ranking: str) -> str:
    """Validates user input fields.
    Returns the canonical ranking string."""
    if not name:
        raise ValueError('Please input your name.')
    if not brought:
        raise ValueError("Please provide your gift's letter code.")
    if (not brought.isalpha()) or (len(brought) > 1):
        raise ValueError('Gift code must be a single letter.')
    if not description:
        raise ValueError('Please provide a description.')
    ranking = ranking.upper()
    for c in ranking:
        if not (c.isalpha() or c.isspace() or (c == '/')):
            raise ValueError(f'Invalid character {c!r} in ranking.')
    if (df['Person'] == name).any():
        raise ValueError(f'Choices already submitted for {name!r}. Please reset data in Google spreadsheet.')
    if (df['Brought'] == brought).any():
        raise ValueError(f'Gift {brought!r} was brought by someone else.')
    items = []
    toks = ranking.strip().split()
    for tok in toks:
        tier = tok.split('/')
        for item in tier:
            if (not item.isalpha()) or (len(item) > 1):
                raise ValueError(f'Error in ranking: {item!r} is not a letter.')
        items.append(tier)
    ranking_obj = Ranking(items)
    if len(ranking_obj) == 0:
        raise ValueError('Please provide your ranking.')
    if len(ranking_obj.universe) < MIN_CHOICES:
        raise ValueError(f'Must rank at least {MIN_CHOICES} items.')
    if SELF_RANKED_LAST and (brought in ranking_obj.universe):
        raise ValueError(f'You may not include your own gift ({brought}) in the ranking.')
    return str(ranking_obj).replace(' ', '')

def process_image(img: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Processes an uploaded image and converts it to a base64 string."""
    thumb = make_thumbnail(img_from_bytes(img.getvalue()))
    return img_to_base64(thumb)


def main() -> None:
    year = datetime.now().year
    st.header(f'Festivus {year} Gift Ranking')
    name = normalize_name(st.text_input('What is your name?', key='name'))
    brought = st.text_input('What is the letter code of the gift you brought?')
    description = st.text_input('Give a brief description of your gift.')
    with st.expander('(Optional) Upload images'):
        st.caption('Try to crop your images to be as "square" as possible.')
        person_img = st.file_uploader('Picture of you', type=IMG_TYPES)
        gift_img = st.file_uploader('Picture of gift you brought', type=IMG_TYPES)
        person_img_str = '' if (person_img is None) else process_image(person_img)
        if person_img_str:
            print(f'Person image: {len(person_img_str)} bytes')
        gift_img_str = '' if (gift_img is None) else process_image(gift_img)
        if gift_img_str:
            print(f'Gift image: {len(gift_img_str)} bytes')
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
    if st.button('Submit', type='primary'):
        df = conn.read(ttl=0).dropna(how='all')
        try:
            ranking = validate_input(df, name, brought, description, ranking)
        except ValueError as e:
            st.error(str(e))
            return
        row = {'Person': name, 'Brought': brought, 'Description': description, 'Ranked Gifts': ranking, 'Person Image': person_img_str, 'Gift Image': gift_img_str}
        df = pd.concat([df, pd.DataFrame.from_records([row])])
        with st.spinner('Submitting data...'):
            st.dataframe(df)
            conn.update(data=df)
        st.write('Your choices have been submitted! 🎉')

main()
