from flask import Flask, render_template, request, jsonify
from model_router import route_and_predict

app = Flask(__name__)


# ===============================
# HOME PAGE
# ===============================
@app.route('/')
def home():
    return render_template('index.html')


# ===============================
# PREDICT API
# ===============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.get('input_data')

        # Convert input string → list
        input_data = [x.strip() for x in data.split(',')]

        # Convert numeric fields safely
        for i in range(len(input_data)):
            try:
                input_data[i] = float(input_data[i])
            except:
                pass  # keep categorical values as string

        result = route_and_predict(input_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "prediction": "Error",
            "model_used": str(e)
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)