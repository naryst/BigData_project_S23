DROP DATABASE IF EXISTS project;

CREATE DATABASE project;

\c project;

CREATE TABLE WATCH_STATUS
(
    status      INTEGER      NOT NULL,
    description VARCHAR(255) NOT NULL,
    PRIMARY KEY (status)
);

CREATE TABLE ANIME
(
    MAL_ID        INTEGER      NOT NULL,
    Name          VARCHAR(255) NOT NULL,
    Score         VARCHAR(255),
    Genres        VARCHAR(255),
    English_Name  VARCHAR(255),
    Japanese_Name VARCHAR(255),
    Type          VARCHAR(255),
    Episodes      VARCHAR(255),
    Aired         VARCHAR(255),
    Premiered     VARCHAR(255),
    Producers     VARCHAR(511),
    Licensors     VARCHAR(255),
    Studios       VARCHAR(255),
    Source        VARCHAR(255),
    Duration      VARCHAR(255),
    Rating        VARCHAR(255),
    Ranked        VARCHAR(255),
    Popularity    INTEGER,
    Members       INTEGER,
    Favorites     INTEGER,
    Watching      INTEGER,
    Completed     INTEGER,
    On_Hold       INTEGER,
    Dropped       INTEGER,
    Plan_To_Watch INTEGER,
    Score_10      VARCHAR(255),
    Score_9       VARCHAR(255),
    Score_8       VARCHAR(255),
    Score_7       VARCHAR(255),
    Score_6       VARCHAR(255),
    Score_5       VARCHAR(255),
    Score_4       VARCHAR(255),
    Score_3       VARCHAR(255),
    Score_2       VARCHAR(255),
    Score_1       VARCHAR(255),
    PRIMARY KEY (MAL_ID)
);

CREATE TABLE ANIME_LIST
(
    user_id          INTEGER NOT NULL,
    anime_id         INTEGER NOT NULL,
    rating           INTEGER NOT NULL,
    watch_status     INTEGER NOT NULL,
    watched_episodes INTEGER NOT NULL
--     CONSTRAINT unique_user_anime UNIQUE (user_id, anime_id),
--     PRIMARY KEY (user_id, anime_id),
--     FOREIGN KEY (anime_id) REFERENCES ANIME(MAL_ID)
--     FOREIGN KEY (watch_status) REFERENCES WATCH_STATUS(status)
);


CREATE TABLE RATINGS
(
    user_id  INTEGER NOT NULL,
    anime_id INTEGER NOT NULL,
    rating   INTEGER NOT NULL,
    PRIMARY KEY (user_id, anime_id),
    FOREIGN KEY (anime_id) REFERENCES ANIME(MAL_ID)
);


-- load data from csv files
\copy watch_status from 'data/watching_status.csv' DELIMITER ',' CSV HEADER;
\copy anime from 'data/anime.csv' DELIMITER ',' CSV HEADER;
\copy ratings from 'data/rating_complete.csv' DELIMITER ',' CSV HEADER;
\copy anime_list from 'data/animelist.csv' DELIMITER ',' CSV HEADER;

DELETE FROM anime_list
WHERE watch_status NOT IN
(SELECT status FROM watch_status);

DELETE FROM anime_list
WHERE anime_id NOT IN
(SELECT mal_id FROM anime);


DELETE FROM anime_list
WHERE (user_id, anime_id) IN (
  SELECT user_id, anime_id
  FROM anime_list
  GROUP BY user_id, anime_id
  HAVING COUNT(*) > 1
)
AND ctid NOT IN (
  SELECT MIN(ctid)
  FROM anime_list
  GROUP BY user_id, anime_id
  HAVING COUNT(*) > 1
);

-- add primary key and foreign keys
ALTER TABLE anime_list
ADD PRIMARY KEY (user_id, anime_id),
ADD CONSTRAINT fk_user_id FOREIGN KEY (watch_status) REFERENCES WATCH_STATUS(status),
ADD CONSTRAINT fk_anime_id FOREIGN KEY (anime_id) REFERENCES anime(mal_id);


-- convert score to float
ALTER TABLE anime
ALTER COLUMN score TYPE FLOAT USING (NULLIF(score, 'Unknown')::FLOAT);


-- convert episodes to integer
ALTER TABLE anime
ALTER COLUMN episodes TYPE INTEGER USING (NULLIF(episodes, 'Unknown')::INTEGER);


-- convert score_10 to integer
ALTER TABLE anime
ALTER COLUMN score_10 TYPE FLOAT USING (NULLIF(score_10, 'Unknown')::FLOAT),
ALTER COLUMN score_9 TYPE FLOAT USING (NULLIF(score_9, 'Unknown')::FLOAT),
ALTER COLUMN score_8 TYPE FLOAT USING (NULLIF(score_8, 'Unknown')::FLOAT),
ALTER COLUMN score_7 TYPE FLOAT USING (NULLIF(score_7, 'Unknown')::FLOAT),
ALTER COLUMN score_6 TYPE FLOAT USING (NULLIF(score_6, 'Unknown')::FLOAT),
ALTER COLUMN score_5 TYPE FLOAT USING (NULLIF(score_5, 'Unknown')::FLOAT),
ALTER COLUMN score_4 TYPE FLOAT USING (NULLIF(score_4, 'Unknown')::FLOAT),
ALTER COLUMN score_3 TYPE FLOAT USING (NULLIF(score_3, 'Unknown')::FLOAT),
ALTER COLUMN score_2 TYPE FLOAT USING (NULLIF(score_2, 'Unknown')::FLOAT),
ALTER COLUMN score_1 TYPE FLOAT USING (NULLIF(score_1, 'Unknown')::FLOAT);
