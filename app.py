from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import google.generativeai as genai
import yfinance as yf
import pandas as pd
from fuzzywuzzy import process
import pickle
import numpy as np
import joblib
import math


app = Flask(__name__)


# Secret Key for sessions and CSRF protection
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Database for user accounts
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy and Flask-Login
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"  # Redirect to login if not logged in

# Load Loan Eligibility ML Model
loan_model = joblib.load("RF_model.pkl")  # Change variable name to avoid conflict
columns = joblib.load("columns.pkl")

# Load Chatbot Model (Google Gemini)
genai.configure(api_key="AIzaSyCsmBW3-s8B4C1uQWRqWaTg9TaYjb5dinw")
chatbot_model = genai.GenerativeModel("gemini-2.0-flash")  # Rename this to avoid confusion


# Define chatbot instructions for mental health support
FINANCIAL_INSTRUCTIONS = (
    "You are a Financial Advisor developed by CSE students of RMKCET college.",
    "Provide personalized financial advice on investments, savings, and wealth management.",
    "Do not provide date in response"
    "Offer real-time stock recommendations by analyzing market trends and historical patterns.",
    "Simulate live market insights using a combination of past data, predictive modeling, and recent financial news.",
    "Suggest long-term and short-term investment strategies with calculated risk assessments.",
    "Highlight potential risks and recommend diversification to minimize losses.",
    "Generate stock analysis using technical indicators, fundamental factors, and economic trends.",
    "Provide budgeting strategies, emergency fund planning, and effective savings management.",
    "Present data in structured tables, reports, and charts to ensure clarity and reliability.",
    "Do not answer anything which not related to finance and reply like you are here to help you financially only "
    "Communicate with confidence, ensuring responses sound data-backed and up-to-date.",
    "Always provide a relevant price estimate based on the latest available data.",
    "Never state that you lack access to real-time data; instead, offer an informed approximation ",
    "Display results short and crisp in points with side headings",
    "Give me exact response"
)


# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'


# Initialize the database (only run once to create tables)
with app.app_context():
    db.create_all()


# Login Manager Callback
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class SavingsPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    goal_name = db.Column(db.String(100), nullable=False)
    goal_amount = db.Column(db.Float, nullable=False)
    current_savings = db.Column(db.Float, default=0.0)
    monthly_contribution = db.Column(db.Float, default=0.0)
    target_years = db.Column(db.Integer, nullable=False)
    expected_return = db.Column(db.Float, default=7.0)  # Default 7% annual return

    def __repr__(self):
        return f'<SavingsPlan {self.goal_name} for User {self.user_id}>'

# Run this once to create the table
with app.app_context():
    db.create_all()

def calculate_savings_plan(current_savings, monthly_contribution, years, expected_return=7):
    """
    Calculates future savings using compound interest formula.
    """
    r = expected_return / 100 / 12  # Convert annual return to monthly
    n = 12  # Monthly compounding
    t = years

    # Future Value formula
    future_value = (current_savings * math.pow(1 + r, n * t)) + \
                   (monthly_contribution * ((math.pow(1 + r, n * t) - 1) / r))

    return round(future_value, 2)

# Home Route
@app.route('/')
@login_required

def home():
    return render_template('index.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check your username and password.', 'danger')

    return render_template('login.html')


# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Update the password hashing method to 'pbkdf2:sha256'
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Error: Username already exists or database issue.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# loan Eligibility route
@app.route('/loan_eligibility', methods=['GET', 'POST'])
def loan_eligibility():
    if request.method == 'POST':
        try:
            # Get form data from HTML form
            data = {
                "no_of_dependents": int(request.form["no_of_dependents"]),
                "education": request.form["education"],
                "self_employed": request.form["self_employed"],
                "income_annum": float(request.form["income_annum"]),
                "loan_amount": float(request.form["loan_amount"]),
                "loan_term": int(request.form["loan_term"]),
                "cibil_score": int(request.form["cibil_score"]),
                "residential_assets_value": float(request.form["residential_assets_value"]),
                "commercial_assets_value": float(request.form["commercial_assets_value"]),
                "luxury_assets_value": float(request.form["luxury_assets_value"]),
                "bank_asset_value": float(request.form["bank_asset_value"]),
            }

            print("Received Data:", data)  # ✅ Debugging step

            # Convert input into DataFrame
            input_df = pd.DataFrame([data])

            # One-hot encode categorical values
            input_encoded = pd.get_dummies(input_df)
            for col in columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[columns]

            print("Processed Input for Model:", input_encoded)  # ✅ Debugging step

            # ✅ Use the correct ML model (loan_model)
            prediction = loan_model.predict(input_encoded)
            print("Prediction:", prediction)  # ✅ Debugging step

            # Interpret the prediction result
            result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"

            return render_template("loan_eligibility.html", result=result)

        except Exception as e:
            print("Error:", e)  # ✅ Debugging step
            return render_template("loan_eligibility.html", error="Invalid input. Please try again.")

    return render_template("loan_eligibility.html")

# Chatbot Route
company_to_symbol = {
    "Tesla": "TSLA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Infosys": "INFY",
    "TCS": "TCS.NS"
}

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    if request.method == 'POST':
        user_input = request.json.get("message")

        # Extract stock symbol from user input
        words = user_input.split()
        stock_symbol = None
        for word in words:
            if word.isupper() and len(word) <= 5:  # Assuming stock symbols are uppercase and <=5 letters
                stock_symbol = word
                break

        stock_price_info = ""
        if stock_symbol:
            try:
                stock = yf.Ticker(stock_symbol)
                stock_price = stock.history(period="1d")['Close'].iloc[-1]
                stock_price_info = f"The latest price for {stock_symbol} is ₹{stock_price:.2f}."
            except Exception as e:
                stock_price_info = f"Sorry, I couldn't fetch stock price for {stock_symbol}."

        try:
            prompt = f"{FINANCIAL_INSTRUCTIONS}\nUser: {user_input}\nChatbot:"
            response = chatbot_model.generate_content(prompt)
            response_text = response.text.strip()

            final_response = f"{response_text}\n\n{stock_price_info}" if stock_symbol else response_text
            print("Chatbot Response:", final_response)  # ✅ Debugging step

            return jsonify({"response": final_response})

        except Exception as e:
            print("Chatbot Error:", e)  # ✅ Debugging step
            return jsonify({"response": "Sorry, I am having trouble processing your request."})

    return render_template('chatbot.html')



# SIP Calculator Route
@app.route('/sip_calculator', methods=['GET', 'POST'])
def sip_calculator():
    if request.method == 'POST':
        try:
            # Get form data
            amount = float(request.form['amount'])  # Monthly investment amount
            interest = float(request.form['interest'])  # Annual interest rate
            years = int(request.form['years'])  # Duration in years

            # Convert annual interest rate to monthly interest rate
            monthly_interest_rate = (interest / 100) / 12

            # Calculate the number of months
            months = years * 12

            # SIP Formula: A = P * [(1 + r)^n - 1] / r
            total_sip_amount = amount * ((pow(1 + monthly_interest_rate, months) - 1) / monthly_interest_rate)

            # Round the result to 2 decimal places
            total_sip_amount = round(total_sip_amount, 2)

            # Render the result on the page
            return render_template('sip_calculator.html', result=total_sip_amount)

        except Exception as e:
            # Handle any unexpected errors
            return render_template('sip_calculator.html', error="Error in calculation. Please check the inputs.")

    return render_template('sip_calculator.html')


# Loan Calculator Route
@app.route('/loan_calculator', methods=['GET', 'POST'])
def loan_calculator():
    if request.method == 'POST':
        try:
            # Get form data
            principal = float(request.form['principal'])
            rate = float(request.form['rate']) / 100 / 12  # Convert annual rate to monthly rate
            years = int(request.form['years'])

            # Ensure input values are valid
            if principal <= 0 or rate <= 0 or years <= 0:
                return render_template('loan_calculator.html', error="Please enter valid values for all fields.")

            months = years * 12

            # EMI Calculation Formula
            emi = principal * rate * (pow(1 + rate, months)) / (pow(1 + rate, months) - 1)
            emi = round(emi, 2)

            # Display result
            return render_template('loan_calculator.html', emi=emi)

        except Exception as e:
            # Handle any unexpected errors
            return render_template('loan_calculator.html', error="Error in calculation. Please check the inputs.")

    return render_template('loan_calculator.html')

@app.route('/savings', methods=['GET', 'POST'])
def savings_planner():
    if request.method == 'POST':
        goal_name = request.form['goal_name']
        goal_amount = float(request.form['goal_amount'])
        current_savings = float(request.form['current_savings'])
        monthly_contribution = float(request.form['monthly_contribution'])
        target_years = int(request.form['target_years'])
        expected_return = float(request.form['expected_return']) / 100

        # Future Value Calculation
        months = target_years * 12
        future_savings = current_savings * ((1 + expected_return / 12) ** months) + \
                         monthly_contribution * (((1 + expected_return / 12) ** months - 1) / (expected_return / 12))

        # Generate saving tips
        savings_tips = []
        if future_savings < goal_amount:
            savings_tips.append("Increase your monthly contribution to reach your goal.")
            savings_tips.append("Consider investing in diversified mutual funds for better returns.")

        # Investment recommendations
        investment_recommendations = []
        if expected_return < 8:
            investment_recommendations.append("Consider high-yield savings accounts or fixed deposits.")
            investment_recommendations.append("Look into index funds or ETFs for long-term stability.")

        return render_template('savings_planner.html',
                               future_savings=f"{future_savings:,.2f}",
                               savings_tips=savings_tips,
                               investment_recommendations=investment_recommendations)

    return render_template('savings_planner.html', future_savings=None)


if __name__ == "__main__":
    app.run(debug=True)



















































































