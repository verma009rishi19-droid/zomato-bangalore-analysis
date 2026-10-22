-- Business questions in SQL (PostgreSQL; also runs on SQLite, which is how the tests check them)

-- Q1. Does a higher price mean a better rating?
SELECT price_band,
       COUNT(*)                AS restaurants,
       ROUND(AVG(rating), 2)   AS avg_rating,
       ROUND(AVG(votes), 0)    AS avg_votes
FROM restaurants
GROUP BY price_band
ORDER BY MIN(cost_for_two);

-- Q2. Online ordering vs rating, all restaurants and then only those with 50+ votes
SELECT online_order, reliable_rating,
       COUNT(*)                    AS restaurants,
       ROUND(AVG(rating), 2)       AS avg_rating,
       ROUND(AVG(cost_for_two), 0) AS avg_cost
FROM restaurants
GROUP BY online_order, reliable_rating
ORDER BY reliable_rating DESC, online_order DESC;

-- Q3. Table booking vs rating (very small group: read with care)
SELECT book_table,
       COUNT(*)              AS restaurants,
       ROUND(AVG(rating), 2) AS avg_rating,
       ROUND(AVG(votes), 0)  AS avg_votes
FROM restaurants
GROUP BY book_table;

-- Q4. Listing type scorecard (a restaurant listed under two types counts in both)
SELECT l.listed_type,
       COUNT(*)                      AS listings,
       ROUND(AVG(r.rating), 2)       AS avg_rating,
       ROUND(AVG(r.cost_for_two), 0) AS avg_cost,
       ROUND(AVG(r.votes), 0)        AS avg_votes
FROM listings l
JOIN restaurants r USING (restaurant_id)
GROUP BY l.listed_type
ORDER BY avg_rating DESC;

-- Q5. How trustworthy are the ratings? Compare thin vs well-reviewed restaurants
SELECT reliable_rating,
       COUNT(*)              AS restaurants,
       ROUND(AVG(rating), 2) AS avg_rating,
       MIN(rating)           AS min_rating,
       MAX(rating)           AS max_rating,
       SUM(CASE WHEN votes = 0 THEN 1 ELSE 0 END) AS zero_vote_restaurants
FROM restaurants
GROUP BY reliable_rating;

-- Q6. Raw rating rank vs vote-adjusted rank (window functions)
SELECT name, rating, votes, bayes_rating,
       RANK() OVER (ORDER BY rating DESC)       AS raw_rank,
       RANK() OVER (ORDER BY bayes_rating DESC) AS adjusted_rank
FROM restaurants
ORDER BY adjusted_rank, name
LIMIT 15;

-- Q7. Great-value picks: 4.0+ rating, 50+ votes, cost at or below the median
WITH med AS (
    SELECT cost_for_two AS median_cost
    FROM (SELECT cost_for_two,
                 ROW_NUMBER() OVER (ORDER BY cost_for_two) AS rn,
                 COUNT(*) OVER ()                          AS n
          FROM restaurants) t
    WHERE rn = (n + 1) / 2
)
SELECT r.name, r.rating, r.votes, r.cost_for_two, r.bayes_rating
FROM restaurants r, med
WHERE r.rating >= 4.0 AND r.votes >= 50 AND r.cost_for_two <= med.median_cost
ORDER BY r.bayes_rating DESC;

-- Q8. Promising but unproven: 4.0+ rating on fewer than 50 votes
SELECT name, rating, votes, cost_for_two
FROM restaurants
WHERE rating >= 4.0 AND votes < 50
ORDER BY votes DESC;

-- Q9. Best well-reviewed restaurant in each price band
SELECT price_band, name, rating, votes, cost_for_two
FROM (
    SELECT price_band, name, rating, votes, cost_for_two,
           ROW_NUMBER() OVER (PARTITION BY price_band ORDER BY bayes_rating DESC, votes DESC) AS rn
    FROM restaurants
    WHERE reliable_rating
) t
WHERE rn = 1
ORDER BY cost_for_two;
