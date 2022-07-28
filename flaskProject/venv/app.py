from flask import Flask, render_template, request, redirect, url_for , jsonify , make_response
import pickle

app = Flask(__name__)
vector_bayes = pickle.load(open('Modelos/count_vector.sav', 'rb'))
modelo_bayes = pickle.load(open('Modelos/naive_bayes.sav', 'rb'))

modelo_KNN = pickle.load(open('Modelos/knn_modelo.sav', 'rb'))

vector_svm = pickle.load(open('Modelos/svm_count_vector.sav', 'rb'))
modelo_svm = pickle.load(open('Modelos/svm_modelo.sav', 'rb'))
# desata la url  pagina raiz

@app.route('/', methods=["GET","POST"])
def main():
    # pickling the vectorizer
    if request.method == "POST":
        inp = request.form.get("inp")
        texto_bayes = vector_bayes.transform([inp])
        sentiment = modelo_bayes.predict(texto_bayes)

        if sentiment[0] == 1:
            return render_template('home.html', message = "Sentimiento Negativo 😔😔😔😔")
        else:
            return render_template('home.html', message = "Sentimiento Positivo 🙂🙂🙂🙂")

    return render_template('home.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
