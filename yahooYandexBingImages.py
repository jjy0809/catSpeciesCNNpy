import os
import requests
import hashlib
from serpapi import GoogleSearch

cat_species = ["Bengal", "Scottish Fold", "British Shorthair", "American Shorthair", "Ragdoll"]
engine_name = ["bing_images", "yahoo_images", "yandex_images"]

downloaded_hashes = set()

def download_images(query, original_query, loop, limit, engine, output_dir=r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img\img", api_key='5fce662fcb840b425490e22a5316ee03da8c7348f92976563822838c4796e487'):
    search = GoogleSearch({
        "engine": engine_name[engine],
        "where": "image",
        "q": query,
        "p": query,
        "query": query,
        "text": query,
        "tbm": "isch",
        "num": limit,
        "api_key": api_key
    })
    results = search.get_dict()

    images = results.get('images_results', [])
    if not images:
        print("No images found.")
        return

    query_dir = os.path.join(output_dir, original_query.replace(" ", "_"))
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)

    for index, image in enumerate(images[:limit]):
        print(f"Try request {index + 1 + limit * loop}")
        try:
            img_url = image["original"]
            img_data = requests.get(img_url, timeout=5).content
            
            print(f"Cheacking {index + 1 + limit * loop}")

            img_hash = hashlib.md5(img_data).hexdigest()

            if img_hash in downloaded_hashes:
                print(f"Skipping duplicate image: {img_url}")
                continue

            print(f"Try Download {index + 1 + limit * loop}")
            
            with open(os.path.join(query_dir, f"{original_query}_{index + 1 + limit * loop}.jpg"), 'wb') as handler:
                handler.write(img_data)
            downloaded_hashes.add(img_hash)
            print(f"Downloaded {original_query}_{index + 1 + limit * loop}.jpg")
        except Exception as e:
            print(f"Could not download {img_url}. Error: {e}")


if __name__ == "__main__":
    search_variations = [
        "kitten",
        "sleeping",
        "outdoor",
        "cute"
    ]
    
    for q in cat_species:
        for e in range(len(engine_name)):
            for loop in range(4): 
                variation = search_variations[loop % len(search_variations)]
                query = f"{q} Cat {variation}".strip()
                download_images(query, q, loop=loop+11+e*4, limit=100, engine=e)

