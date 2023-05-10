psql -U postgres -f sql/postgre.sql

hdfs dfs -rm -r /project
rm -r /project/avsc

sqoop import-all-tables     -Dmapreduce.job.user.classpath.first=true     --connect jdbc:postgresql://localhost/project     --username postgres     --warehouse-dir /project     --as-avrodatafile     --compression-codec=snappy     --outdir /project/avsc     --m 1

