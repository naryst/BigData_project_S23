USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY 'q5'
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT COUNT(DISTINCT anime_id), COUNT(DISTINCT user_id)
FROM anime_list;