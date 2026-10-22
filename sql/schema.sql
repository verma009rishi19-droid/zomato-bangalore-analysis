-- PostgreSQL schema for the cleaned Zomato data
DROP TABLE IF EXISTS listings;
DROP TABLE IF EXISTS restaurants;

CREATE TABLE restaurants (
    restaurant_id    INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    online_order     BOOLEAN NOT NULL,
    book_table       BOOLEAN NOT NULL,
    rating           NUMERIC(2,1) NOT NULL,
    votes            INTEGER NOT NULL,
    cost_for_two     INTEGER NOT NULL,
    price_band       TEXT NOT NULL,
    bayes_rating     NUMERIC(3,2) NOT NULL,    -- rating shrunk toward the mean for low vote counts
    reliable_rating  BOOLEAN NOT NULL,         -- true when votes >= 50
    listed_types     TEXT
);

CREATE TABLE listings (
    restaurant_id INTEGER REFERENCES restaurants (restaurant_id),
    listed_type   TEXT NOT NULL,
    PRIMARY KEY (restaurant_id, listed_type)
);
