import os
import requests
import hashlib
from serpapi import GoogleSearch


cat_species = ["Korean Shorthair", "Persian", "Siamese", "Maine Coon", "Russian Blue", "Bengal", "Scottish Fold", "British Shorthair", "American Shorthair", "Ragdoll"]

downloaded_hashes = set()

def download_images(query, original_query, loop, limit=100, output_dir=r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img", api_key='17ebb6a62ca9d291b543651214b9aa82d348a070e5bc83e695624a9df15dfe41'):
    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        "num": limit,
        "start": loop * limit,
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
        "",
        "cute",
        "kitten",
        "sleeping",
        "outdoor",
        "face"
    ]
    
    for q in cat_species:
        for loop in range(6):  
            variation = search_variations[loop % len(search_variations)]
            query = f"{q} Cat {variation}".strip()
            download_images(query, q, loop=loop, limit=100)
