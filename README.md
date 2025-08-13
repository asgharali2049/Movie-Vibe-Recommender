# 🎬 Movie Vibe Recommender
## 📌 Overview
The **Movie Vibe Recommender** is a web application that suggests movies based on the *vibe* and *genre* of a movie you input.  
It uses **TF-IDF vectorization** and **cosine similarity** to analyze movie descriptions and recommend similar ones.  
---
## ✨ Features

- 🔍 Search for a movie by name.
- 🎯 Get recommendations based on the movie’s description and genre.
- 📊 Uses natural language processing for text-based similarity.
- 🌐 Simple and interactive **Streamlit** UI.
---
## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** – Data handling  
- **Scikit-learn** – TF-IDF Vectorizer & Cosine Similarity  
- **Streamlit** – Web app interface
---
## 📂 Project Structure

- movie-vibe-recommender/
- │-- app.py # Main Streamlit app
- │-- movies.csv # Dataset with movie titles, genres, and descriptions
- │-- requirements.txt # Python dependencies
- │-- README.md # Project documentation
---
## 🚀 Installation & Usage

1. **Clone the repository**
   ```
   git clone https://github.com/your-username/movie-vibe-recommender.git
   cd movie-vibe-recommender```
2. **Install dependencies**
```pip install -r requirements.txt```
3. **Run the app**
```streamlit run main.py```
4. **Open in browser**
 Streamlit will provide a local URL (usually http://localhost:8501).

## 📊 How It Works
1. Loads the movie dataset.
2. Converts movie descriptions into numerical vectors using TF-IDF.
3. Calculates cosine similarity between the searched movie and all others.
4. Returns the top n most similar movies
