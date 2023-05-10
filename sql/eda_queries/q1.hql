USE projectdb;
INSERT OVERWRITE LOCAL DIRECTORY 'q1'
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
SELECT user_id, AVG(rating) AS mean_rating
FROM anime_list
GROUP BY user_id
ORDER BY mean_rating DESC;

