USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY 'q2'
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT watch_status.description, COUNT(*) AS count
FROM anime_list
INNER JOIN watch_status ON anime_list.watch_status = watch_status.status
GROUP BY watch_status.description
ORDER BY count DESC;

