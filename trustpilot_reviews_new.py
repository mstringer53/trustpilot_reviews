# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:09:34 2025

@author: mstri
"""

import streamlit as st
import requests
import time
import datetime
import sqlite3
import hashlib
from bs4 import BeautifulSoup
import pandas as pd

import pyodbc

import lxml

import numpy as np

from requests import get

import emoji
import html5lib

import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from deep_translator import GoogleTranslator 
from langdetect import detect


import os




# Database setup
DB_NAME = "review_db___.sqlite"



WIKI_COUNTRY_URL = "https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#Officially_assigned_code_elements"

def initialize_database():
    """Creates necessary tables if they do not already exist, including country codes."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # **Metadata Table for Last Scraped Page**
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            brand TEXT PRIMARY KEY,
            last_scraped_page INTEGER,
            total_pages INTEGER,
            last_scraped_date TEXT
        )
    """)

    # **Raw Reviews Table (Unprocessed Data)**
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            author_id TEXT,
            location TEXT,
            num_reviews TEXT,
            review_title TEXT,
            review_txt TEXT,
            review_date TEXT,
            experience_date TEXT,
            review_score TEXT,
            invited_review TEXT
        )
    """)

    # **Cleaned Reviews Table (Processed Data)**
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews_clean (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            author_id TEXT,
            location TEXT,
            num_reviews TEXT,
            review_title TEXT,
            review_txt TEXT,
            review_title_clean TEXT,
            review_txt_clean TEXT,
            review_title_trans TEXT,
            review_trans TEXT,
            review_text_clean TEXT,
            review_date TEXT,
            experience_date TEXT,
            review_score TEXT,
            invited_review TEXT,
            country_name
        )
    """)

    # **Country Codes Table**
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS country_codes (
            country_code TEXT PRIMARY KEY,
            country_name TEXT
        )
    """)
    
    # Track each page that has been scraped
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_pages (
            brand TEXT,
            page_number INTEGER,
            scraped_date TEXT,
            PRIMARY KEY (brand, page_number)
        )
    """)

    

    conn.commit()

    # **Debugging: Check if Table Exists**
    #cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    #tables = cursor.fetchall()
    #print("✅ Tables in DB:", tables)  # This will show all existing tables

    conn.close()

    


def populate_country_codes():
    """Scrapes and stores country codes in the database if not already populated."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # **Check if data already exists**
    cursor.execute("SELECT COUNT(*) FROM country_codes")
    if cursor.fetchone()[0] > 0:
        #print("Country codes already exist in the database.")
        conn.close()
        return

    # **Scrape country codes from Wikipedia**
    tables = pd.read_html(WIKI_COUNTRY_URL)
    cc_table = tables[4]
    cc_table = cc_table.iloc[:, [0, 1]].copy()  # Keep only the first 2 columns
    cc_table.columns = ["country_code", "country_name"]
    cc_table["country_code"] = cc_table["country_code"].str.lower()  # Lowercase for matching

    # **Insert into database**
    cc_table.to_sql("country_codes", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()    
    #print("Country codes stored in the database.")



# Ensure Database and Tables Exist Before Running Streamlit
initialize_database()  # Creates `reviews_raw`, `reviews_clean`, `metadata`, `country_codes`
populate_country_codes()  # Fetch country codes only if needed



if "stop_scraping" not in st.session_state:
    st.session_state.stop_scraping = False  # Default to False (meaning: continue scraping)

def get_last_page_number(company):
    """Scrapes Trustpilot to get the last available review page number."""
    url = f"https://uk.trustpilot.com/review/{company}?languages=all&sort=recency"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    last_page_element = soup.select("nav a")  # Extracts pagination elements

    if last_page_element:
        try:
            return int(last_page_element[-2].text.strip())  # Gets last numbered page
        except ValueError:
            return 1  # If only 1 page exists, return 1
    return 1


def get_last_scraped_page(company):
    """Fetches last scraped page & total pages from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT last_scraped_page, total_pages FROM metadata WHERE brand = ?", (company,))
    result = cursor.fetchone()
    conn.close()
    return result if result else (None, None)


def update_last_scraped_page(company, last_page, total_pages):
    """Updates the database with the last scraped page and total pages."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO metadata (brand, last_scraped_page, total_pages, last_scraped_date)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(brand) DO UPDATE SET last_scraped_page = ?, total_pages = ?, last_scraped_date = ?
    """, (company, last_page, total_pages, datetime.datetime.now().strftime('%Y-%m-%d'),
          last_page, total_pages, datetime.datetime.now().strftime('%Y-%m-%d')))
    conn.commit()
    conn.close()


def determine_scrape_pages(company):
    """Determines the correct range of pages to scrape, ensuring we always scrape from the oldest reviews first."""
    total_pages = get_last_page_number(company)
    last_scraped_page, prev_total_pages = get_last_scraped_page(company)

    # **Case 1: First-time scrape (No data in metadata)**
    if last_scraped_page is None:
        #start_page = max(1, total_pages - 49)  # Start from the last available page downwards
        #end_page = total_pages  # Scrape from the last page backwards
        start_page = total_pages  # Start at the highest page
        end_page = max(1, total_pages - 49)  # Scrape downward in batches of 50
        
        return start_page, end_page, total_pages

    # **Case 2: Scraping again (new pages exist)**
    #new_pages = total_pages - prev_total_pages if prev_total_pages else 0

    # **Adjust logic to ensure we scrape from the oldest pages that haven't been collected**
    #start_page = max(1, last_scraped_page - 49)
    #end_page = last_scraped_page
    start_page = last_scraped_page
    end_page = max(1, last_scraped_page - 49)

    return start_page, end_page, total_pages



def generate_author_id(author, location, brand):
    """Generate a hashed unique ID for author + location + brand."""
    unique_str = f"{author}_{location}_{brand}"
    return hashlib.sha256(unique_str.encode()).hexdigest()[:7]  # Shorter 7-char hash


# Emoji Processing
def process_emojis(text):
    """ 
    - If text is only emojis, convert them to word equivalents.
    - If text contains a mix of text and emojis, remove emojis.
    - If text is None, return an empty string.
    """
    if isinstance(text, str):
        stripped_text = emoji.replace_emoji(text, replace='')  # Remove emojis

        if stripped_text.strip():  # If non-empty after removal, return without emojis
            return stripped_text
        else:  # If it was only emojis, convert to words
            return emoji.demojize(text).replace(":", " ").replace("_", " ")

    return ""  # Return empty string if not a string


# Text Translation
def eng_translate(text):
    """ 
    - Translates text if not in English, with a delay to prevent throttling 
    """
    if pd.isnull(text) or text.strip() == "":
        return text
    try:
        if detect(text) != 'en':
            time.sleep(0.2)  # Adjust delay if needed
            return GoogleTranslator(source='auto', target='english').translate(text)
        return text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # Return original text if translation fails


def map_country_names():
    """
    - Updates `country_name` in `reviews_clean` by looking up country_codes based on `location`
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Perform an SQL JOIN to update country names
    cursor.execute("""
        UPDATE reviews_clean
        SET country_name = (
            SELECT country_name 
            FROM country_codes 
            WHERE LOWER(country_codes.country_code) = LOWER(reviews_clean.location)
        )
        WHERE country_name IS NULL OR country_name = ''
    """)
    # If no match is found, set country_name = location
    cursor.execute("""
        UPDATE reviews_clean
        SET country_name = location
        WHERE (country_name IS NULL OR country_name = '') AND location != ''
    """)
    

    conn.commit()
    conn.close()
    #print("Country names updated in reviews_clean table.")

# Run this after processing new reviews
map_country_names()


def scrape_trustpilot(company, start_page, end_page):
    """Scrapes reviews from Trustpilot within the selected page range."""
    reviews = []
    
    # 🚀 DEBUG: Check if function is called
    print(f"🔍 scrape_trustpilot() called with Start: {start_page}, End: {end_page}")
    
    # progress bar to track which pages are being scraped
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_pages = end_page - start_page + 1
    
    # Open DB connection once (rather than opening & closing in each loop)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    for idx, p in enumerate(range(start_page, end_page + 1)):
        print(f"✅ Scraping page {p}...")  # 🚀 DEBUG LINE
        url = f"https://uk.trustpilot.com/review/{company}?languages=all&page={p}&sort=recency"
        time.sleep(.5)  # Prevents excessive requests
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        data = soup.find_all('article', {'data-service-review-card-paper': "true"})
        
        # Log the scraped page
        cursor.execute("""
            INSERT OR IGNORE INTO scraped_pages (brand, page_number, scraped_date) 
            VALUES (?, ?, ?)
        """, (company, p, datetime.datetime.now().strftime('%Y-%m-%d')))
        
        
        # Update progress bar text to show the current page being scraped
        status_text.text(f"Scraping page {p} of {end_page}...")
        
        

        for container in data:
            author = container.find('span', {'data-consumer-name-typography': "true"})
            location = container.find('div', {'data-consumer-country-typography': "true"})
            review_title = container.find('a', {'data-review-title-typography': "true"})
            review_text = container.find('p', {'data-service-review-text-typography': "true"})
            review_score = container.find('div', {'class': "styles_reviewHeader__xV2js"})
            time_element = container.find('time', {'data-service-review-date-time-ago': "true"})
            experience_timestamp = container.find('p', {'data-service-review-date-of-experience-typography': "true"})

            author = author.text.strip() if author else ''
            location = location.text.strip() if location else ''
            review_title = review_title.text.strip() if review_title else ''
            review_text = review_text.text.strip() if review_text else ''
            review_score = review_score.get("data-service-review-rating") if review_score else ''
            review_date = time_element.get('datetime').split('T')[0] if time_element else ''
            experience_date = experience_timestamp.text.split(": ")[-1] if experience_timestamp else ''
            
            # Capture `num_reviews`
            t_review = getattr(container.find('span', {'data-consumer-reviews-count-typography': "true"}), 'text', None) or ''
            num_reviews = t_review.rpartition(" ")[0]  # Extract numeric part

            # Capture `invited_review`
            invited_review = getattr(container.find('button', {'data-review-label-tooltip-trigger': "true"}), 'text', None) or ''

            author_id = generate_author_id(author, location, company)

            reviews.append((company, author_id, location, num_reviews, review_title, review_text, review_date, experience_date, review_score, invited_review))
        
        progress_bar.progress((idx +1)/total_pages)

    # Store in DB
    #conn = sqlite3.connect(DB_NAME)
    #cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO reviews_raw (brand, author_id, location, num_reviews, review_title, review_txt, review_date, experience_date, review_score, invited_review)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, reviews)
    conn.commit()
    conn.close()

    return len(reviews)  # Number of reviews scraped

def process_reviews():
    """Cleans and translates reviews, moving them from 'reviews_raw' to 'reviews_clean'."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Get unprocessed reviews including `num_reviews` and `invited_review`
    cursor.execute("""
        SELECT id, brand, author_id, location, num_reviews, review_title, review_txt, review_date, 
               experience_date, review_score, invited_review 
        FROM reviews_raw
    """)
    rows = cursor.fetchall()
    
    total_reviews = len(rows)  # Get the total number of reviews to process
    if total_reviews == 0:
        return 0  # No reviews to process

    processed_reviews = []
    progress_bar = st.progress(0)  # Initialize the progress bar
    
    for idx, row in enumerate(rows):
        review_id, brand, author_id, location, num_reviews, review_title, review_txt, review_date, experience_date, review_score, invited_review = row

        # Process emojis
        review_title_clean = process_emojis(review_title)
        review_txt_clean = process_emojis(review_txt)

        # Translate if necessary
        review_title_trans = eng_translate(review_title_clean)
        review_trans = eng_translate(review_txt_clean)

        # Use review_trans first, fallback to review_title_trans
        review_text_clean = review_trans if review_trans.strip() else review_title_trans

        if review_text_clean.strip():  # **Only store if text is not empty**
            processed_reviews.append((brand, author_id, location, num_reviews, review_title, review_txt,
                                      review_title_clean, review_txt_clean, review_title_trans, review_trans,
                                      review_text_clean, review_date, experience_date, review_score, invited_review))

        # Delete from `reviews_raw` after processing
        cursor.execute("DELETE FROM reviews_raw WHERE id = ?", (review_id,))
        
        # **Update progress bar**
        progress_bar.progress((idx + 1) / total_reviews)
        

    # Store cleaned data in `reviews_clean`
    cursor.executemany("""
        INSERT INTO reviews_clean (brand, author_id, location, num_reviews, review_title, review_txt, 
                                  review_title_clean, review_txt_clean, review_title_trans, review_trans, 
                                  review_text_clean, review_date, experience_date, review_score, invited_review)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, processed_reviews)

    conn.commit()
    conn.close()
    
    map_country_names()
    
    # **Complete the progress bar**
    progress_bar.empty()  # Remove the progress bar after completion

    return len(processed_reviews)



# Define session state for stopping scraping
if "stop_scraping" not in st.session_state:
    st.session_state.stop_scraping = False


def scrape_trustpilot_auto(company, start_page, end_page, batch_size=50):
    """Automatically scrapes reviews in batches while allowing user to stop."""
    
    total_pages = end_page - start_page + 1
    progress_bar = st.progress(0)
    status_text = st.empty()

    while start_page >= 1 and not st.session_state.stop_scraping:
        batch_end_page = max(1, start_page - batch_size + 1)  # Move backward in batches
        pages_scraped = scrape_trustpilot(company, batch_end_page, start_page)

        # ✅ **Store progress in the database**
        
        
        num_processed = process_reviews()
        update_last_scraped_page(company, batch_end_page, end_page)
        
        
        # ✅ **Update UI messages**
        st.success(f"✔ Scraped {pages_scraped} reviews from pages {batch_end_page} → {start_page}!")
        st.write(f"✅ Moving to next batch: {batch_end_page - batch_size} → {batch_end_page}...")
        st.success(f"🔍 Processed {num_processed} reviews successfully!")

        start_page -= batch_size  # Move to the next batch

    st.success("✅ Scraping complete!")


# ** STREAMLIT APP **
st.title("Trustpilot Review Scraper")


# User inputs the company to scrape
company = st.text_input("Enter Trustpilot URL segment (e.g., `www.trustedhousesitters.com`):")

#st.write(f"📂 Database Location: {os.path.abspath(DB_NAME)}")

if company:
    start_page, end_page, last_page = determine_scrape_pages(company)
    
    # Get previously scraped pages
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT page_number) FROM scraped_pages WHERE brand = ?", (company,))
    stored_reviews = cursor.fetchone()[0] or 0
    conn.close()

    st.write(f" **Total pages available:** {last_page}")
    st.write(f" **Total pages already scraped:** {stored_reviews}")
    st.write(f" **Recommended scrape range:** {start_page} → {end_page}")
    
    scrape_option = st.radio("Select Scrape Option:", [
        "Scrape all available pages",
        "Scrape a custom range",
        "Scrape the next 50 pages"
    ])

    if scrape_option == "Scrape a custom range":
        custom_start_page = st.number_input("Start Page:", min_value=1, max_value=last_page, value=start_page)
        custom_end_page = st.number_input("End Page:", min_value=1, max_value=last_page, value=end_page)
        start_page = custom_end_page
        end_page = custom_start_page
        
    if st.button("Stop Scraping"):
        st.session_state.stop_scraping = True #Flag to stop scraping
        st.warning("Scraping will stop after the current batch is complete")
        
        # 🚀 DEBUGGING: Check values before calling scrape function
        st.write(f"🔍 DEBUG: Start Page = {start_page}, End Page = {end_page}")
        
    if st.button("Start Scraping"):
        st.session_state.stop_scraping = False #reset stop flag
        
        

        if scrape_option == "Scrape all available pages":
            pages_scraped = scrape_trustpilot_auto(company, start_page, end_page, batch_size=50)
        elif scrape_option == "Scrape a custom range":
            st.write("✅ Running scrape_trustpilot() for custom range")  # 🚀 DEBUG LINE
            pages_scraped = scrape_trustpilot(company, start_page, end_page)
        else:
            pages_scraped = scrape_trustpilot(company, max(end_page - 50, 1),end_page)  # Scrape next 50 pages

        
        # **Process reviews immediately after scraping**
        num_processed = process_reviews()
        update_last_scraped_page(company, end_page, last_page)

        st.success(f"✅ Scraped {pages_scraped} reviews from pages {start_page} → {end_page}!")
        st.success(f"🔍 Processed {num_processed} reviews successfully!")


    

