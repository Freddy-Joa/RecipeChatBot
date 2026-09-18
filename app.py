import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

print("Loading recipes...")

data = pd.read_csv(r"C:\Users\louro\Downloads\RAW_recipes.csv.zip", compression="zip")

# Use smaller sample for speed
data = data.sample(15000, random_state=42)

# Combine ingredients
data["ingredients_text"] = data["ingredients"].astype(str)

# Build AI search model
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X = vectorizer.fit_transform(data["ingredients_text"])

print("Recipe engine ready!")

@app.route("/", methods=["GET","POST"])
def home():

    recipes = []

    if request.method == "POST":

        user_input = request.form["ingredients"]

        user_vec = vectorizer.transform([user_input])

        similarity = cosine_similarity(user_vec, X)

        top_index = similarity.argsort()[0][-5:][::-1]

        for i in top_index:

            recipe = {
                "name": data.iloc[i]["name"],
                "ingredients": data.iloc[i]["ingredients"],
                "steps": data.iloc[i]["steps"]
            }

            recipes.append(recipe)

    return render_template("index.html", recipes=recipes)


if __name__ == "__main__":
    app.run(debug=True)