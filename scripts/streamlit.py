import altair as alt
import streamlit as st
import pandas as pd

Q2_df = pd.read_csv("output/q2.csv")

Hist1 = (
    alt.Chart(Q2_df)
    .mark_bar()
    .encode(x="watch_status", y="anime_count", color="watch_status")
    .properties(width=800, height=600)
)

st.title("Anime Watch Status")
st.write(Hist1)
st.write(
    "Since we need to greatly truncate the data for training, "
    "we decide to use Completed, Currently watching "
    "and Dropped watch status rows for our model. "
    "Logically, such data is more reliable. "
)
# ----------------------------------------------
Q4_df = pd.read_csv("output/q4.csv")

Names = Q4_df.columns.tolist()
Values = Q4_df.iloc[0].tolist()

New_q4_df = pd.DataFrame({"names": Names, "values": Values})

Bar = (
    alt.Chart(New_q4_df)
    .mark_bar()
    .encode(
        x=alt.X("names", sort=None),
        y="values",
        color="names",
        tooltip=["names", "values"],
    )
    .properties(width=800, height=600)
)

st.title("Score counts")
st.write(Bar)
st.write("From this graph we can see that score users give to the anime.  "
         "That we can consider as a bad and good score for the anime. "
         "We will use rating >= 5 as a relevant recommendation.")
# ----------------------------------------------
Q5_df = pd.read_csv("output/q5.csv")

Names = Q5_df.columns.tolist()
Values = Q5_df.iloc[0].tolist()

New_q5_df = pd.DataFrame({"names": Names, "values": Values})

Hist2 = (
    alt.Chart(New_q5_df)
    .mark_bar()
    .encode(
        x=alt.X("names", sort=None),
        y="values",
        color="names",
        tooltip=["names", "values"],
    )
    .properties(width=800, height=600)
)
st.title("Unique users and animes")
st.write(Hist2)
st.write("From this histogram we can see, how much unique"
         " users and unique animes we have in the table anime_list. ")
# ----------------------------------------------
Q1_df = pd.read_csv("output/q1.csv")

# Create the Altair chart
Chart = (
    alt.Chart(Q1_df)
    .mark_bar()
    .encode(x=alt.X("mean_rating", bin=alt.Bin(step=0.5)), y="count()")
    .properties(width=800, height=600)
)

# Show the chart using Streamlit
st.title("Mean rating distribution among all users")
st.write(Chart)
st.write("For this graph we initially counted mean rating among all animes, "
         "what user watched for all users.  And then we plot distribution of the"
         " users and their mean ratings. As we can see at zero we have data "
         "anomaly. Whats happened because significant part of users just "
         "add animes in the list and dont rate them. "
         "We have reset this unrated values to zero. So, thats why this happened")
# ----------------------------------------------
Q6_df = pd.read_csv("output/q6.csv")

# filter out users with more than 1000 ratings
# q6_df = q6_df[q6_df['ratings_count'] < 1100]
Chart = (
    alt.Chart(Q6_df)
    .mark_bar()
    .encode(
        x=alt.X("ratings_count", scale=alt.Scale(domain=(0, 18000))),
        y="count()",
    )
    .properties(width=800, height=600)
)

st.title("Rating count distribution among all users")
st.write(Chart)
st.write(" In this graph we are showing, distribution of how much animes "
         "user have watched and rated. We took about 2000 unique users "
         "with the highest number of rated anime. The cutoff "
         "is approximately ~1500.")
# ----------------------------------------------

Q7_df = pd.read_csv("output/q7.csv")
# columns anime_id, votes_count
# plot distribution of votes_count
Chart = (
    alt.Chart(Q7_df)
    .mark_bar()
    .encode(
        x=alt.X("votes_count", scale=alt.Scale(domain=(0, 250000))),
        y="count()",
    )
    .properties(width=800, height=600)
)
st.title("Votes count distribution among all animes")
st.write(Chart)
st.write("In this graph we counted number of votes for "
         "each anime in the dataset. This can show us "
         "an information, which anime we should include in "
         "our final dataset. Unpopular animes will not "
         "give us sensible performance  boosting, but will "
         "increase complexity of the model alot. "
         "We took about 3000 unique anime with the "
         "highest rated numbers by users. "
         "The cutoff is approximately ~5000.")

############# Metrics of the models #############
Metrics = ["model", "MAP@5", "NDCG@5"]
data = [
    ["Baseline", 0.00309840306841, 0.24899296692],
    ["ALS model", 0.315916460769, 0.589849124161],
    ["ALS after tuning", 0.308796228157, 0.505887802758],
]

Metrics_df = pd.DataFrame(data, columns=Metrics)

st.title("Metrics of the models")

Style = {"font-size": "25px"}
Metrics_df = Metrics_df.style.set_properties(**Style)
st.table(Metrics_df)

