import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from trail import get_response

load_dotenv()

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    bot_response = ""
    if request.method == "POST":
        user_message = request.form["message"]
        bot_response = get_response(user_message)
    return render_template("index.html", response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)
