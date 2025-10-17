from flask import Flask, request, jsonify
from chatbot_model import get_response  # Import the get_response function from your chatbot_model.py

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({'error': 'Invalid Content-Type. Must be application/json'}), 415
    
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    response = get_response(question)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
