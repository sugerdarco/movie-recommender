import sys
from core.recommender import Recommender

def main():
    rec = Recommender()
    rec.build()

    # movie ID can be passed from command line or fallback to default
    if len(sys.argv) > 1:
        movie_id = sys.argv[1]
    else:
        movie_id = "m001" # default

    print(f"\n🔍 Recommendations for movie_id: {movie_id}\n")

    try:
        recommendations = rec.recommend(movie_id, top_k=5)
        for r in recommendations:
            print(f"- {r['title']} ({r['movie_id']}) | Score: {r['score']:.4f}")
    except ValueError as e:
            print(f"{e}")

if __name__ == "__main__":
    main()
