rm -rf output/q*
rm -rf q*

hive -f sql/eda_queries/q1.hql 

echo "User_id,mean_rating" > output/q1.csv
cat q1/* >> output/q1.csv

hive -f sql/eda_queries/q2.hql

echo "watch_status,anime_count" > output/q2.csv
cat q2/* >> output/q2.csv


hive -f sql/eda_queries/q3.hql

echo "mean_rating" > output/q3.csv
cat q3/* >> output/q3.csv


hive -f sql/eda_queries/q4.hql

echo "score_1,score_2,score_3,score_4,score_5,score_6,score_7,score_8,score_9,score_10" > output/q4.csv
cat q4/* >> output/q4.csv


hive -f sql/eda_queries/q5.hql

echo "unique_anime,unique_user" > output/q5.csv
cat q5/* >> output/q5.csv

hive -f sql/eda_queries/q6.hql

echo "user_id,ratings_count" > output/q6.csv
cat q6/* >> output/q6.csv

hive -f sql/eda_queries/q7.hql

echo "anime_id,votes_count" > output/q7.csv
cat q7/* >> output/q7.csv

rm -rf q*
