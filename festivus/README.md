# Festivus

A Google Sheet has been set up to store people's rankings. See [these instructions](https://docs.streamlit.io/develop/tutorials/databases/private-gsheet).

- [Link to Google Sheet][Google sheet]
- [Streamlit Apps](https://share.streamlit.io)
  - [Submission app]
  - [SMP app]

## Submission App

The Festivus submission app, `festivus_submit.py`, is running on [Streamlit][Submission app].

Submissions populate the [Google Sheet].

Validation performed:

- Required: name, gift brought (single letter), description, ranking of at least 5 items.
- Name cannot already have an entry in the spreadsheet.
- Cannot rank your own gift.

## SMP App

The SMP (ranking) app, `smp_app.py`, is running on [Streamlit][SMP app].

This lets you upload a CSV file of gift rankings, and it will produce an animated video of the Gale-Shapley matching procedure.

## Step-by-step Guide for Festivus

1. Gather all of the gifts in one place. Label them with single-letter codes.
2. Share the submission app [link][Submission app] with all participants.
3. All participants submit their rankings.

<!-- Links -->

[Google sheet]: https://docs.google.com/spreadsheets/d/1Ofsf4TvR66I3hSslhSe2dR--cN-ZaeA31Mkb7blzgqo/edit?gid=0#gid=0
[Submission app]: https://cppwbuzhkpdjhnnjfgjvge.streamlit.app
[SMP app]: https://jeremander-pysoc-binsmp-app-m1nk2v.streamlit.app
