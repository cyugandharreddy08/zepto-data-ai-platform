import os
import re
import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Constants
BASE_URL = 'http://books.toscrape.com/'
EXCHANGE_RATE_GBP_TO_INR = 105.50  # Fixed baseline conversion rate: 1 GBP = 105.50 INR
DB_PATH = os.path.join(os.path.dirname(__file__), 'zepto_catalog.db')
RATING_MAP = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5
}

CATEGORIES_TO_SCRAPE = [
    ('Travel', 'catalogue/category/books/travel_2/index.html'),
    ('Mystery', 'catalogue/category/books/mystery_3/index.html'),
    ('Historical Fiction', 'catalogue/category/books/historical-fiction_4/index.html'),
    ('Sequential Art', 'catalogue/category/books/sequential-art_5/index.html'),
]

def scrape_category(category_name, rel_url):
    books = []
    current_url = BASE_URL + rel_url
    
    while current_url:
        resp = requests.get(current_url, timeout=10)
        if resp.status_code != 200:
            print(f'Failed to fetch {current_url}: status {resp.status_code}')
            break
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        pods = soup.select('article.product_pod')
        
        for pod in pods:
            title_tag = pod.h3.a
            title = title_tag['title'] if title_tag.has_attr('title') else title_tag.text.strip()
            
            price_tag = pod.select_one('p.price_color')
            raw_price = price_tag.text.strip() if price_tag else None
            
            rating_tag = pod.select_one('p.star-rating')
            raw_rating = None
            if rating_tag:
                classes = [c.lower() for c in rating_tag.get('class', []) if c != 'star-rating']
                raw_rating = classes[0] if classes else None
                
            avail_tag = pod.select_one('p.instock.availability')
            raw_avail = avail_tag.text.strip() if avail_tag else None
            
            books.append({
                'title': title,
                'raw_price': raw_price,
                'raw_rating': raw_rating,
                'raw_avail': raw_avail,
                'category': category_name
            })
            
        # Check next page within category
        next_tag = soup.select_one('li.next a')
        if next_tag:
            next_page = next_tag['href']
            # Reconstruct URL relative to current directory of category
            curr_dir = current_url.rsplit('/', 1)[0]
            current_url = f'{curr_dir}/{next_page}'
        else:
            current_url = None
            
    return books

def clean_data(raw_books):
    df = pd.DataFrame(raw_books)
    print(f'Total raw books scraped: {len(df)}')
    
    # 1. Clean Price
    def parse_price(val):
        if not val or pd.isna(val):
            return None
        cleaned = re.sub(r'[^\d.]', '', str(val))
        try:
            return float(cleaned)
        except ValueError:
            return None
            
    df['price_gbp'] = df['raw_price'].apply(parse_price)
    
    # Impute missing price using category median or global median
    if df['price_gbp'].isna().any():
        print('Imputing missing price_gbp with category median...')
        df['price_gbp'] = df.groupby('category')['price_gbp'].transform(lambda x: x.fillna(x.median()))
        df['price_gbp'] = df['price_gbp'].fillna(df['price_gbp'].median())
        
    # 2. Rating
    def parse_rating(val):
        if not val or pd.isna(val):
            return 3  # Median rating default if missing
        return RATING_MAP.get(str(val).lower(), 3)
        
    df['rating'] = df['raw_rating'].apply(parse_rating).astype(int)
    
    # 3. Availability
    def parse_availability(val):
        if not val or pd.isna(val):
            return False
        return 'in stock' in str(val).lower()
        
    df['in_stock'] = df['raw_avail'].apply(parse_availability).astype(int)
    
    # 4. Currency conversion using fixed rate (1 GBP = 105.50 INR)
    df['price_inr'] = (df['price_gbp'] * EXCHANGE_RATE_GBP_TO_INR).round(2)
    
    # Drop rows without title
    df = df.dropna(subset=['title']).copy()
    
    cleaned_df = df[['title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category']].copy()
    print(f'Cleaned dataset shape: {cleaned_df.shape}')
    return cleaned_df

def init_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    ''')
    
    cur.execute('''
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    ''')
    
    conn.commit()
    return conn

def load_to_sqlite(df, conn):
    cur = conn.cursor()
    # Insert categories
    unique_categories = df['category'].unique()
    for cat in unique_categories:
        cur.execute('INSERT OR IGNORE INTO categories (category_name) VALUES (?)', (cat,))
    conn.commit()
    
    # Fetch category_id map
    cat_map = dict(cur.execute('SELECT category_name, category_id FROM categories').fetchall())
    
    # Prepare book rows
    books_data = []
    for _, row in df.iterrows():
        books_data.append((
            row['title'],
            row['price_gbp'],
            row['price_inr'],
            row['rating'],
            row['in_stock'],
            cat_map[row['category']]
        ))
        
    cur.executemany('''
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', books_data)
    conn.commit()
    print(f'Successfully loaded {len(books_data)} books across {len(cat_map)} categories into SQLite.')

def run_queries(conn):
    queries = [
        (
            "Query 1: SELECT & WHERE - High-rated books with 5-star rating",
            """
            SELECT title, price_gbp, price_inr, rating 
            FROM books 
            WHERE rating = 5 
            LIMIT 5;
            """
        ),
        (
            "Query 2: ORDER BY & LIMIT - Top 5 most expensive books in INR",
            """
            SELECT title, price_inr, rating 
            FROM books 
            ORDER BY price_inr DESC 
            LIMIT 5;
            """
        ),
        (
            "Query 3: DISTINCT - Distinct star ratings present in catalog",
            """
            SELECT DISTINCT rating 
            FROM books 
            ORDER BY rating ASC;
            """
        ),
        (
            "Query 4: BETWEEN & IN - Books priced between 30 and 40 GBP with rating in (4, 5)",
            """
            SELECT title, price_gbp, price_inr, rating 
            FROM books 
            WHERE price_gbp BETWEEN 30.0 AND 40.0 
              AND rating IN (4, 5)
            LIMIT 5;
            """
        ),
        (
            "Query 5: JOIN - Top 10 highest-rated books with category details",
            """
            SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating 
            FROM books b 
            JOIN categories c ON b.category_id = c.category_id 
            WHERE b.rating = 5 
            ORDER BY b.price_inr DESC 
            LIMIT 10;
            """
        )
    ]
    
    results = {}
    print('\n' + '='*70)
    print('EXECUTING SQL QUERIES AGAINST NORMALIZED SQLITE DATABASE')
    print('='*70)
    for name, sql in queries:
        print(f'\n--- {name} ---')
        print(f'SQL:\n{sql.strip()}')
        res_df = pd.read_sql(sql, conn)
        print('Output:')
        print(res_df.to_string(index=False))
        results[name] = res_df
        
    return results

def verify_pandas_equivalence(conn):
    print('\n' + '='*70)
    print('VERIFYING EQUIVALENCE: pd.read_sql vs pd.merge')
    print('='*70)
    
    # 1. Read directly via SQL Join
    sql_join = """
    SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating 
    FROM books b 
    JOIN categories c ON b.category_id = c.category_id 
    WHERE b.rating = 5 
    ORDER BY b.price_inr DESC 
    LIMIT 10;
    """
    df_sql = pd.read_sql(sql_join, conn)
    
    # 2. Replicate in-memory using pd.merge
    books_df = pd.read_sql('SELECT * FROM books', conn)
    categories_df = pd.read_sql('SELECT * FROM categories', conn)
    
    merged = pd.merge(books_df, categories_df, on='category_id', how='inner')
    filtered = merged[merged['rating'] == 5]
    sorted_df = filtered.sort_values(by='price_inr', ascending=False).head(10)
    df_pandas = sorted_df[['book_id', 'title', 'category_name', 'price_gbp', 'price_inr', 'rating']].reset_index(drop=True)
    
    print('SQL Join Result (pd.read_sql):')
    print(df_sql)
    print('\nIn-Memory Merge Result (pd.merge):')
    print(df_pandas)
    
    # Assertion of equivalence
    pd.testing.assert_frame_equal(df_sql, df_pandas, check_dtype=False)
    print('\nSUCCESS: SQL Join and in-memory pd.merge produced IDENTICAL outputs!')
    return df_sql, df_pandas

def main():
    print('Starting Zepto Data Pipeline...')
    all_books = []
    for cat_name, rel_url in CATEGORIES_TO_SCRAPE:
        print(f'Scraping category: {cat_name}...')
        cat_books = scrape_category(cat_name, rel_url)
        print(f'  Fetched {len(cat_books)} items.')
        all_books.extend(cat_books)
        
    cleaned_df = clean_data(all_books)
    conn = init_db(DB_PATH)
    load_to_sqlite(cleaned_df, conn)
    run_queries(conn)
    verify_pandas_equivalence(conn)
    conn.close()
    print(f'\nData Pipeline completed successfully. Database persisted at: {DB_PATH}')

if __name__ == '__main__':
    main()
