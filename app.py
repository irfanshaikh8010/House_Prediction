from flask import Flask, request, render_template
import numpy as np
import joblib

obj = joblib.load("california3.joblib")

model = obj["model"]
columns = obj["columns"]

print("Columns:", columns)

app = Flask(__name__)


@app.route("/")
def main():
    return render_template(
        "index.html",
        columns=columns
    )


@app.route("/predict", methods=["POST"])
def predict():

    Input = []

    try:

        for i in columns:

            val = request.form.get(i, type=float)

            if val is None:
                return render_template(
                    "index.html",
                    columns=columns,
                    error=f"Please enter a valid value for {i}"
                )

            Input.append(val)

        Input = np.array(Input).reshape(1, -1)

        prediction = model.predict(Input)

        result = prediction[0]

        return render_template(
            "index.html",
            columns=columns,
            prediction=result
        )

    except Exception as e:

        return render_template(
            "index.html",
            columns=columns,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)