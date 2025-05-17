# link tutorial : https://youtu.be/OwxxCibSFKk?si=ogX9zwnWNzlFzfxp
# repo 1 : https://github.com/TheRobBrennan/explore-docker-python-flask-nextjs-typescript/tree/main
# repo 2 : https://github.com/martindavid/flask-nextjs-user-management-example

from flask import Flask, jsonify
from holt_winter.hw_analysis import run_hw_analysis

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Flask Holt-Winter API is running!"})

@app.route("/run-analysis", methods=["GET"])
def run_analysis():
    try:
        run_hw_analysis()
        return jsonify({"message": "Holt-Winter analysis completed and saved to database."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
