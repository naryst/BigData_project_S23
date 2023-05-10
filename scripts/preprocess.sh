#!/bin/bash
echo "================installing python dependecies================"
pip install -r requirements.txt
echo "================installing data from google drive as .zip archive================"
cd data/
gdown 1amxJ_plPdc3_a9pOxdhuY1C1ZyrV3v-C 
echo "================extracting from archive================"
unzip anime-recommendation-database-2020.zip

