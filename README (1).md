
# Stock Prediction AI

This project features a sophisticated stock prediction application composed of the following components:
1. **`app.py`**: The backend Python script responsible for processing data and generating predictions.
2. **`historical_analyses.html`**: The webpage dedicated to displaying historical analyses of stock predictions.
3. **`stock-prediction-website.html`**: The primary user interface for inputting stock symbols and viewing predictions.

## Setup

### Prerequisites
Ensure the following libraries are installed:

#### Python Libraries
- Flask
- pandas
- numpy
- scikit-learn
- tensorflow
- requests

Install the required Python libraries with the following command:
```bash
pip install Flask pandas numpy scikit-learn tensorflow requests
```

#### HTML Libraries
The HTML files leverage the following libraries, included via CDN:
- [Chart.js](https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.min.js)
- [TensorFlow.js](https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.11.0/dist/tf.min.js)
- [Bootstrap](https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css)
- [jsPDF](https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.3.1/jspdf.umd.min.js)

### Running the Application

1. **Backend Server**: Launch the backend server by executing `app.py`, which will start a Flask server:
```bash
python app.py
```

2. **Frontend**: Open `stock-prediction-website.html` in your web browser. This is the main interface for interacting with the application.

## Usage

1. **Input Stock Symbol**: Enter the stock symbol (e.g., AAPL) in the input field and click "Analyze" to initiate the prediction process.
2. **View Results**: The results, including historical data, predictions, technical analysis, sentiment analysis, recommendations, and accuracy, will be displayed.
3. **View Historical Analyses**: Click "View Historical Analyses" to access past analyses stored in `historical_analyses.html`.
4. **Download Report**: Generate and download the analysis report as a PDF by clicking "Download Report".

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

This project was developed by Filippo Roffilli.
