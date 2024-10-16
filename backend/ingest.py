import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from backend.vectorstore import create_vector_store
import tiktoken
import time

from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
import mysql.connector
from bs4 import BeautifulSoup
import re
from openpyxl import Workbook
from Levenshtein import distance as levenshtein_distance
import hashlib

embeddings = OpenAIEmbeddings()

def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()


def is_valid_content(text):
    if text.strip().startswith('a:'):
        return False
    if text.count('{') > 5:
        return False
    return True


def normalize_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s]', '', text)
    return text


def clean_and_validate_content(content):
    cleaned = clean_html(content)
    normalized = normalize_text(cleaned)
    
    valid_lines = [line for line in normalized.split('\n') if is_valid_content(line)]
    
    return '\n'.join(valid_lines)


def get_wordpress_content(host, user, password, database):
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = conn.cursor()

    query = "SELECT DISTINCT post_content FROM wp_posts WHERE post_status = 'publish'"
    # query = "SELECT DISTINCT post_content FROM wp_posts"
    cursor.execute(query)

    posts = cursor.fetchall()
    
    cursor.close()
    conn.close()

    return posts


def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_content(content, min_chunk_size=100, max_chunk_size=1000):
    chunks = []
    paragraphs = content.split('\n')
    
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += " " + paragraph
    
    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks


def process_wordpress_content(host, user, password, database, min_chunk_size=50, max_chunk_size=1500, output_file='chunks.xlsx'):
    start_time = time.time()
    posts = get_wordpress_content(host, user, password, database)
    all_chunks = []
    processed_hashes = {}

    total_posts = len(posts)
    print(f"Total posts to process: {total_posts}")

    for index, post in enumerate(posts, 1):
        if index % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {index}/{total_posts} posts. Elapsed time: {elapsed_time:.2f} seconds")

        content = clean_html(post[0])
        
        # Check if the content starts with 'a:' before normalization
        if content.strip().startswith('a:'):
            print(f"Skipping post {index}: Invalid content (starts with 'a:')")
            continue
        
        normalized_content = normalize_text(content)
        
        if len(normalized_content) < min_chunk_size:
            print(f"Skipping post {index}: Content too short")
            continue
        
        content_hash = hash_text(normalized_content)
        
        # Check for exact duplicates
        if content_hash in processed_hashes:
            print(f"Skipping post {index}: Exact duplicate")
            continue
        
        # Check for near duplicates
        is_duplicate = False
        for existing_hash, existing_text in processed_hashes.items():
            if abs(len(normalized_content) - len(existing_text)) <= 20:
                if levenshtein_distance(normalized_content, existing_text) <= 10:
                    is_duplicate = True
                    break
        
        if is_duplicate:
            print(f"Skipping post {index}: Near duplicate")
            continue
        
        processed_hashes[content_hash] = normalized_content
        chunks = chunk_content(content, min_chunk_size, max_chunk_size)
        all_chunks.extend(chunks)
        print(f"Processed post {index}: Added {len(chunks)} chunks")

    print(f"Processing complete. Total chunks: {len(all_chunks)}")

    print("Writing chunks to Excel file...")
    wb = Workbook()
    ws = wb.active
    ws.title = "Chunks"
    ws.append(["Chunk"])

    for chunk in all_chunks:
        ws.append([chunk])

    wb.save(output_file)

    total_time = time.time() - start_time
    print(f"Chunks have been written to {output_file}")
    print(f"Total processing time: {total_time:.2f} seconds")

    return all_chunks


def calculate_embedding_cost(chunks):
    """Calculates the embedding cost for a list of string chunks using the OpenAI Ada 002 tokenizer."""
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum(len(encoding.encode(chunk)) for chunk in chunks)
    cost = (total_tokens / 1000) * 0.0004
    return cost


def main():
    index_name = 'ferry-chatbot'

    host = 'localhost'
    user = 'rag'
    password = 'ragapp'
    database = 'keltaslt'

    chunks = process_wordpress_content(host, user, password, database)

    print(f"Number of chunks: {len(chunks)}")
    print(calculate_embedding_cost(chunks))

    create_vector_store(index_name, chunks)


if __name__ == '__main__':
    main()
