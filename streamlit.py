import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt

q2_df = pd.read_csv('output/q2.csv')

hist = alt.Chart(q2_df).mark_bar().encode(
    x='watch_status',
    y='anime_count',
    color='watch_status'
).properties(
    width=800,
    height=600
)

st.title('Anime Watch Status')
st.write(hist)

q4_df = pd.read_csv('output/q4.csv')

names = q4_df.columns.tolist()
values = q4_df.iloc[0].tolist()

new_q4_df = pd.DataFrame({
    'names': names,
    'values': values
})

bar = alt.Chart(new_q4_df).mark_bar().encode(
    x=alt.X('names', sort=None),
    y='values',
    color='names',
    tooltip=['names', 'values'],
).properties(
    width=800,
    height=600
)

st.title('Score counts')
st.write(bar)

q5_df = pd.read_csv('output/q5.csv')

names = q5_df.columns.tolist()
values = q5_df.iloc[0].tolist()

new_q5_df = pd.DataFrame({
    'names': names,
    'values': values
})

hist = alt.Chart(new_q5_df).mark_bar().encode(
    x=alt.X('names', sort=None),
    y='values',
    color='names',
    tooltip=['names', 'values'],
).properties(
    width=800,
    height=600
)
st.title('Unique users and animes')
st.write(hist)

q1_df = pd.read_csv('output/q1.csv')

# Create the Altair chart
chart = alt.Chart(q1_df).mark_bar().encode(
    x=alt.X('mean_rating', bin=alt.Bin(step=0.5)),
    y='count()'
).properties(
    width=800,
    height=600
)

# Show the chart using Streamlit
st.title('Mean rating distribution among all users')
st.write(chart)

q6_df = pd.read_csv('output/q6.csv')

chart = alt.Chart(q6_df).mark_bar().encode(
    x=alt.X('ratings_count', scale=alt.Scale(domain=(0, 18000))),
    y='count()',
).properties(
    width=800,
    height=600
)

st.title('Rating count distribution among all users')
st.write(chart)

############# Metrics of the models #############
metrics = ['model', 'MAP@5', 'NDCG@5']
data = [['Baseline', 0.00309840306841, 0.24899296692],
        ['ALS model', 0.315916460769, 0.589849124161],
        ['ALS after tuning', 0.308796228157, 0.505887802758]]

metrics_df = pd.DataFrame(data, columns=metrics)

st.title('Metrics of the models')

style = {'font-size': '25px'}
metrics_df = metrics_df.style.set_properties(**style)
st.table(metrics_df)

