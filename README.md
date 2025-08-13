# 🎬 Movie Vibe Recommender
## 📌 Overview
The **Movie Vibe Recommender** is a web application that suggests movies based on the *vibe* and *genre* of a movie you input.  
It uses **TF-IDF vectorization** and **cosine similarity** to analyze movie descriptions and recommend similar ones.  

<img width="1876" height="803" alt="Screenshot 2025-08-12 162206" src="https://github.com/user-attachments/assets/74a220bd-c30c-414f-821c-a68299fe156c" />

## ✨ Features
- 🔍 Movie Search – Find similar movies by entering a title
- 🎯 Vibe Matching – TF-IDF + Cosine Similarity to find description/genre-based matches
- 🎬 Director-Based Filtering – Get recommendations from the same director for a more consistent style & storytelling
- 🎭 Genre-Based Filtering – Discover movies from the same or related genres for mood-specific suggestions
- 🤝 Item-Based Recommendations – Suggests movies based on similarity to a chosen movie (content-based approach)
- 🌐 Interactive Streamlit UI – Minimal, fast, and user-friendly
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
---
## 📊 How It Works
1. Loads the movie dataset.
2. Converts movie descriptions into numerical vectors using TF-IDF.
3. Calculates cosine similarity between the searched movie and all others.
4. Returns the top n most similar movies
---
## 📜 License
This project is licensed under the MIT License – free to use and modify.

---
## 🤝 Contributing
Pull requests are welcome! Please open an issue first to discuss changes.

---
## “Let your next favorite movie find you.” 🌟
   
