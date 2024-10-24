from flask import Flask, request, jsonify
from recommend import get_recommendations
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)


# Define a route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend_courses():
    user_data = request.get_json()
    feedback = user_data.get('feedback', '')
    
    if not feedback:
        return jsonify({'error': 'Feedback is required'}), 400
    
    recommendations = get_recommendations(feedback)
    return jsonify({'recommendations': recommendations})

# Define a health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server is running'})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
