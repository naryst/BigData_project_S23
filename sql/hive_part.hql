DROP DATABASE IF EXISTS projectdb CASCADE;
CREATE DATABASE projectdb;
USE projectdb;
SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;
CREATE EXTERNAL TABLE anime STORED AS AVRO LOCATION '/project/anime'
TBLPROPERTIES ('avro.schema.url'='/project/avsc/anime.avsc');
CREATE EXTERNAL TABLE anime_list STORED AS AVRO LOCATION '/project/anime_list'
TBLPROPERTIES ('avro.schema.url'='/project/avsc/anime_list.avsc');
CREATE EXTERNAL TABLE ratings STORED AS AVRO LOCATION '/project/ratings'
TBLPROPERTIES ('avro.schema.url'='/project/avsc/ratings.avsc');
CREATE EXTERNAL TABLE watch_status STORED AS AVRO LOCATION '/project/watch_status'
TBLPROPERTIES ('avro.schema.url'='/project/avsc/watch_status.avsc');
