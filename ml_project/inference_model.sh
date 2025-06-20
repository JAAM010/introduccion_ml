#!/bin/bash


curl -X 'POST' \
  'http://localhost:8890/invocations' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "dataframe_split": {
  "columns": [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity"
  ],
  "data": [
    [
      -122.23,
      37.88,
      41.0,
      880.0,
      129.0,
      322.0,
      126.0,
      8.3252,
      "NEAR BAY"
    ]
  ]
  }
}'
