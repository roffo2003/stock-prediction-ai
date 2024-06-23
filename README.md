# stock-prediction-ai
Stock Prediction AI leverages machine learning to predict stock prices. It includes a backend for data processing and a frontend for inputting stock symbols and viewing predictions, technical analyses, sentiment analyses, and recommendations. Features real-time predictions, technical and sentiment analysis, and historical data storage.


Stock Prediction AI
This project features a sophisticated stock prediction application composed of the following components:

app.py: The backend Python script responsible for processing data and generating predictions.
historical_analyses.html: The webpage dedicated to displaying historical analyses of stock predictions.
stock-prediction-website.html: The primary user interface for inputting stock symbols and viewing predictions.
Setup
Prerequisites
Ensure the following libraries are installed:

Python Libraries
Flask
pandas
numpy
scikit-learn
tensorflow
requests
Install the required Python libraries with the following command:

bash
Copia codice
pip install Flask pandas numpy scikit-learn tensorflow requests
HTML Libraries
The HTML files leverage the following libraries, included via CDN:

Chart.js
TensorFlow.js
Bootstrap
jsPDF
Running the Application
Backend Server: Launch the backend server by executing app.py, which will start a Flask server:
bash
Copia codice
python app.py
Frontend: Open stock-prediction-website.html in your web browser. This is the main interface for interacting with the application.
Usage
Input Stock Symbol: Enter the stock symbol (e.g., AAPL) in the input field and click "Analyze" to initiate the prediction process.
View Results: The results, including historical data, predictions, technical analysis, sentiment analysis, recommendations, and accuracy, will be displayed.
View Historical Analyses: Click "View Historical Analyses" to access past analyses stored in historical_analyses.html.
Download Report: Generate and download the analysis report as a PDF by clicking "Download Report".
License
This project is licensed under the MIT License. See the LICENSE file for details.

Author
This project was developed by Filippo Roffilli.

