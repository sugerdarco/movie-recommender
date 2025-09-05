import sys
from core.recommender import Recommender


# Function to compute/build embeddings and index
def initialize_recommender():
    rec = Recommender()
    rec.build()  # compute embeddings and build HNSW index
    return rec


# Function to handle user input and give recommendations
def query_recommendations(rec, movie_id=None):
    if not movie_id:
        movie_id = "m001"  # default movie ID

    print(f"\n Recommendations for movie_id: {movie_id}\n")
    try:
        recommendations = rec.recommend(movie_id, top_k=5)
        for r in recommendations:
            print(f"- {r['title']} ({r['movie_id']}) | Score: {r['score']:.4f}")
    except ValueError as e:
        print(f"{e}")

    # Optional loop for additional queries
    while True:
        movie_id = input("\nEnter another movie ID (or 'exit' to quit): ").strip()
        if movie_id.lower() == "exit":
            break
        try:
            recommendations = rec.recommend(movie_id, top_k=5)
            print(f"\n🔍 Recommendations for movie_id: {movie_id}\n")
            for r in recommendations:
                print(f"- {r['title']} ({r['movie_id']}) | Score: {r['score']:.4f}")
        except ValueError as e:
            print(f"{e}")


# Main function
def main():
    # Step 1: Compute/build embeddings and index once
    rec = initialize_recommender()

    # Step 2: Take initial movie ID from command line
    movie_id = sys.argv[1] if len(sys.argv) > 1 else None
    query_recommendations(rec, movie_id)

if __name__ == "__main__":
    main()
