-- Zepto Data Pipeline - Analytical SQL Queries
-- Database: zepto_catalog.db

-- Query 1: SELECT & WHERE
-- Filters 5-star rated books
SELECT title, price_gbp, price_inr, rating 
FROM books 
WHERE rating = 5 
LIMIT 5;

-- Query 2: ORDER BY & LIMIT
-- Retrieves top 5 most expensive books in INR
SELECT title, price_inr, rating 
FROM books 
ORDER BY price_inr DESC 
LIMIT 5;

-- Query 3: DISTINCT
-- Lists all unique star ratings in the catalog
SELECT DISTINCT rating 
FROM books 
ORDER BY rating ASC;

-- Query 4: BETWEEN & IN
-- Retrieves books priced between 30 and 40 GBP with ratings of 4 or 5
SELECT title, price_gbp, price_inr, rating 
FROM books 
WHERE price_gbp BETWEEN 30.0 AND 40.0 
  AND rating IN (4, 5)
LIMIT 5;

-- Query 5: Two-Table JOIN
-- Joins books with categories to list the top 10 highest-rated books by INR price
SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating 
FROM books b 
JOIN categories c ON b.category_id = c.category_id 
WHERE b.rating = 5 
ORDER BY b.price_inr DESC 
LIMIT 10;
