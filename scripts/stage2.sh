hdfs dfs -rm /project/avsc/*
hdfs dfs -mkdir /project/avsc

hdfs dfs -put /project/avsc/*.avsc /project/avsc

hive -f sql/hive_part.hql

sh scripts/EDA.sh

