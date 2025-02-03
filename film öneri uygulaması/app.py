import requests
import pandas as pd
from flask import Flask, request, render_template


api_key = '7bfd2e5d8c9f8e03d6c8a8b2cd8f41c0'
url = f'https://api.themoviedb.org/3/movie/popular?api_key={api_key}&language=en-US&page=1'
headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3YmZkMmU1ZDhjOWY4ZTAzZDZjOGE4YjJjZDhmNDFjMCIsIm5iZiI6MTcyOTYxNzgyMC42MTY5LCJzdWIiOiI2NzE3ZGNmOGUwM2E0M2MzNjlhYzczZWMiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.5Nv5vs4PhNi7ce6BQAsG3NPyrUyAgNqDRPdqbtb6ghI"
}
response = requests.get(url,headers=headers)
data = response.json()
print(response.text)
print(data)


if response.ok:
    data = response.json()
else:
    print("API çağrısında hata:", response.status_code)
    data = {'results': []}  # Boş sonuç döndür



# TMDB'den aldığın veri
movies = pd.DataFrame(data['results'])

# Türleri liste haline getir ve dataframe'e ekle
movies['genres'] = movies['genre_ids'].apply(lambda x: ', '.join(map(str, x)))
print(movies.head())



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Film açıklamalarını TF-IDF ile vektörize et
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Cosine Similarity hesapla
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    index = indices[title]
    sim_scores = list(enumerate(cosine_sim[index]))

    

    sim_scores = sorted(sim_scores, key =lambda x:x[1], reverse=True)


    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',movies=pd.DataFrame())

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['movie']
    recommendations = get_recommendations(movie)
    
    if recommendations.empty:
        return render_template('index.html', error="Bu film için öneri bulunamadı.")

    return render_template('index.html', movies=recommendations)

if __name__ == '__main__':
    app.run(debug=True)


