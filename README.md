# House Price Prediction with Decision Tree

A machine learning web application that predicts house prices using Decision Tree Regression.

## 🎯 Features
- Interactive slider inputs for house characteristics
- Real-time price predictions
- Feature importance visualization
- Price breakdown analysis
- Responsive design

## 🏠 Input Features
- **Size**: House area in square feet (800-3500 sq ft)
- **Bedrooms**: Number of bedrooms (1-6)
- **Age**: Property age in years (0-50 years)
- **Distance**: Distance from city center in km (1-30 km)

## 🚀 How to Run Locally

1. Clone the repository:
```bash
   git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
   cd house-price-predictor
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
   streamlit run app.py
```

4. Open your browser at `http://localhost:8501`

## 📊 Model Performance
- **Algorithm**: Decision Tree Regressor
- **Training R² Score**: ~0.95
- **Test R² Score**: ~0.90
- **Average Prediction Error**: ~$50,000

## 🔧 Technical Details
- **Framework**: Streamlit
- **ML Library**: Scikit-learn
- **Visualization**: Plotly
- **Language**: Python 3.8+

## 📈 Model Training
The model was trained on a synthetic dataset containing:
- 500 house samples
- 4 features (Size, Bedrooms, Age, Distance)
- Realistic price relationships

## 🎓 Project Purpose
Created for Lab 09 - Machine Learning course to demonstrate:
- Decision Tree Regression implementation
- Model deployment with Streamlit
- Interactive ML application development
- GitHub hosting and version control

## 📝 License
Educational project - Free to use and modify

## 👨‍💻 Author
[Your Name]
Lab 09 - Decision Tree Assignment
