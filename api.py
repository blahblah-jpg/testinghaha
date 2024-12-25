from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/endpoint', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        array = np.array(data, dtype=np.float32)  # Ensure the array is of type FLOAT32
        interpreter = tf.lite.Interpreter("model.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], array)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        return jsonify(output_data.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run("0.0.0.0",5000)