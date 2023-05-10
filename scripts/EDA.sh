rm -rf output/q*
rm -rf q*

hive -f sql/eda_queries/q1.hql 

echo "User_id,mean_rating" > output/q1.csv
cat q1/* >> output/q1.csv

hive -f sql/eda_queries/q2.hql

echo "watch_status,anime_count" > output/q2.csv
cat q2/* >> output/q2.csv

rm -rf q*
