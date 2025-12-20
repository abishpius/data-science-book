# Chapter 7: Predictive Modeling with Linear Regression

Introduction to Scikit-Learn and the Modeling Workflow
Up until this point in the book, we have functioned primarily as historians. We have cleaned historical records, visualized past trends, and used statistical tests to determine if past events were significant. We have been answering the question: What happened?
Now, we cross the threshold into the most exciting part of Data Science: Predictive Modeling. We are shifting our focus from explaining the past to predicting the future.
To do this, we will move beyond standard Python arithmetic and Pandas manipulation. We will introduce the industry-standard library for machine learning in Python: Scikit-Learn (often shortened to sklearn).
The Machine Learning Paradigm Shift
Before we write code, we must adjust our mental model. In traditional programming (like the cleaning scripts we wrote in Chapter 3), you provide the computer with Rules and Data, and it gives you Answers.
* Traditional Programming: If Marketing_Spend > 1000, then Label = "High Priority".
In Machine Learning, we flip this. We provide the computer with Data and the Answers (historical results), and we ask the computer to learn the Rules.
* Machine Learning: Here is how much we spent on marketing last year, and here is how much revenue we made. You tell me the mathematical relationship between them.
The Vocabulary of Prediction
To use Scikit-Learn effectively, you must become comfortable with its specific terminology. You will see these terms used in documentation, StackOverflow answers, and job interviews.
1. Target ($y$): This is what you are trying to predict. It is the "Answer." In a spreadsheet, this is usually a single column. (e.g., Revenue, House Price, Customer Churn). 2. Features ($X$): These are the variables you use to make the prediction. These are the "Inputs." (e.g., Marketing Spend, Square Footage, Number of Customer Support Calls). 3. Model: The mathematical engine that learns the relationship between $X$ and $y$. 4. Training: The process of letting the model look at your data to learn the rules.
 A diagram showing a standard Excel-style dataset. The last column is highlighted in Red and labeled "Target (y) - What we want to predict". The first three columns are highlighted in Blue and labeled "Features (X) - The data we use to predict". An arrow points from X to y labeled "The Model learns this relationship". 

A diagram showing a standard Excel-style dataset. The last column is highlighted in Red and labeled "Target (y) - What we want to predict". The first three columns are highlighted in Blue and labeled "Features (X) - The data we use to predict". An arrow points from X to y labeled "The Model learns this relationship".
Note on Notation: In Python and Data Science conventions, we use a capital $X$ for features because it represents a matrix (multiple columns/dimensions), and a lowercase $y$ for the target because it represents a vector (a single column/dimension).
Introducing Scikit-Learn
Scikit-Learn is the most popular machine learning library for Python. It is open-source, robust, and incredibly consistent. The beauty of Scikit-Learn is its API consistency. Whether you are performing a simple Linear Regression (fitting a straight line) or a complex Random Forest (an ensemble of decision trees), the Python syntax remains nearly identical.
Once you learn the "Scikit-Learn Workflow," you can apply it to almost any algorithm.
The 5-Step Modeling Workflow
Every supervised machine learning project in Scikit-Learn follows this specific recipe. We will walk through the concepts first, and then apply them to code.
Step 1: Arrange Data into Features ($X$) and Target ($y$) We must separate our DataFrame. We slice out the column we want to predict and save it as y. We select the columns we want to use for prediction and save them as X.
Step 2: Train/Test Split This is the most critical concept for avoiding "cheating."
Imagine you are teaching a student (the Model) for a math exam. You give them a textbook containing 100 practice questions and the answers at the back. If the student memorizes the answers to all 100 questions, they will score 100% on a test if the test uses those exact same questions. However, if you give them a new question, they will fail. They didn't learn the math; they memorized the data. This is called Overfitting.
To prevent this, we hide a portion of the data. 1. Training Set (e.g., 80% of data): The model is allowed to see this. It uses this to learn. 2. Testing Set (e.g., 20% of data): The model never sees this during training. We hold it back to evaluate how well the model performs on "unseen" data.
 A visual representation of the Train/Test Split. A horizontal bar representing a dataset is cut into two pieces. The larger piece (80%) is colored Green and labeled "Training Set (Model learns from this)". The smaller piece (20%) is colored Orange and labeled "Testing Set (Used to evaluate performance)". A "No Peeking!" icon separates the two. 

A visual representation of the Train/Test Split. A horizontal bar representing a dataset is cut into two pieces. The larger piece (80%) is colored Green and labeled "Training Set (Model learns from this)". The smaller piece (20%) is colored Orange and labeled "Testing Set (Used to evaluate performance)". A "No Peeking!" icon separates the two.
Step 3: Instantiate the Model We create an instance of the algorithm we want to use. In this chapter, we are using LinearRegression. Think of this as opening an empty box that has the capacity to learn, but hasn't learned anything yet.
Step 4: Fit the Model This is the magic step. We command the model to "Fit" or "Train" on the Training Data. The model looks at the $X_{train}$ and compares it to the $y_{train}$ to calculate the best mathematical formula.
Step 5: Predict and Evaluate Once the model is trained, we give it the $X_{test}$ (the exam questions without answers) and ask it to predict the $y$. We then compare its predictions against the actual $y_{test}$ values to grade its performance.
Implementation in Python
Let's apply this workflow to a simulated dataset. Imagine we are analyzing a retail business and want to predict Sales based on Marketing Budget.
First, let's set up our data.
```python
import pandas as pd


# Creating a sample dataset
data = {
    'Marketing_Budget': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    'Sales': [22000, 25000, 29000, 35000, 42000, 46000, 50000, 56000, 61000, 65000]
}


df = pd.DataFrame(data)


# Display the first few rows
print(df.head(3))
```

Now, we follow the 5-step workflow using Scikit-Learn.
Step 1: Separate X and y Note the double brackets [['Marketing_Budget']] for X. Scikit-Learn expects X to be a DataFrame (2D), even if it only has one column.
```python
# Step 1: Define Features (X) and Target (y)
```

X = df[['Marketing_Budget']] # Features (Capital X, 2D array)
y = df['Sales']              # Target (Lowercase y, 1D array)
Step 2: The Train/Test Split We use the train_test_split utility from Scikit-Learn. We will specify test_size=0.2 (holding back 20% of the data) and a random_state (to ensure your split looks the same as mine every time you run the code).
```python
from sklearn.model_selection import train_test_split


# Step 2: Split the data
```

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
Step 3 & 4: Instantiate and Fit We import the LinearRegression class. This algorithm attempts to draw the "Line of Best Fit" through our data (we will explore the math of this line in the next section).
```python
from sklearn.linear_model import LinearRegression


# Step 3: Instantiate the model
lr_model = LinearRegression()


# Step 4: Fit the model (The Learning Phase)
# NOTE: We only fit on the TRAINING data!
lr_model.fit(X_train, y_train)


print("Model has been trained.")
```

Step 5: Predict Now that lr_model has learned the relationship between budget and sales, we can ask it to predict the sales for our test set, or even for a completely new budget number.
```python
# Step 5: Make predictions
# Let's predict the sales for the Test set (which the model hasn't seen)
predictions = lr_model.predict(X_test)


# Let's look at the results side-by-side
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results)


# We can also predict for a hypothetical budget of $12,000
new_budget = [[12000]] # Double brackets for 2D shape
future_prediction = lr_model.predict(new_budget)
print(f"Predicted sales for $12k budget: ${future_prediction[0]:,.2f}")
[[IMAGE: A flowchart summarizing the code block above. 1. Box "Raw Data" splits into -> "X (Features)" and "y (Target)". 2. Arrows lead to "Train/Test Split". 3. "Train Set" goes into "Model.fit()". 4. "Test Set" goes into "Model.predict()". 5. The output of predict and the actual Test labels meet at "Evaluation/Comparison".]]
```

Summary You have just built your first Machine Learning pipeline. While the dataset was simple, the process is identical to what Data Scientists use at Google, Netflix, or Amazon:
1. Isolate what you want to predict ($y$). 2. Split your data to prevent overfitting. 3. Initialize a model. 4. Train the model on the training set. 5. Use the model to make predictions.
In the next section, we will lift the hood of the LinearRegression model to understand exactly how it calculated those predictions and how to interpret the "Line of Best Fit" for business stakeholders.
Simple Linear Regression: Forecasting Continuous Variables
In the previous section, we introduced Scikit-Learn and established the predictive modeling workflow: Instantiate, Fit, Predict. We are now ready to apply this workflow to the most fundamental algorithm in Data Science: Simple Linear Regression.
While "Linear Regression" sounds like a dry statistical term, in a business context, it is a "Crystal Ball generator." It allows us to move from saying "Marketing and Sales are correlated" (Descriptive) to saying "If we increase the marketing budget by $1,000, Sales will increase by exactly $4,200" (Predictive).
The Geometry of Prediction
At its core, Simple Linear Regression attempts to fit a straight line through your data points that best represents the relationship between two variables: 1. The Independent Variable ($X$): The input or driver (e.g., Marketing Spend). 2. The Dependent Variable ($y$): The output or target (e.g., Revenue).
Imagine we have a scatter plot of last year's marketing campaigns.
 A scatter plot with 'Marketing Spend ($)' on the x-axis and 'Revenue ($)' on the y-axis. The data points show a positive trend, moving upward from left to right, indicating that as spend increases, revenue increases. 

A scatter plot with 'Marketing Spend ($)' on the x-axis and 'Revenue ($)' on the y-axis. The data points show a positive trend, moving upward from left to right, indicating that as spend increases, revenue increases.
Our goal is to draw a line through these dots. However, we cannot just draw any line; we need the "best" line. But what defines "best"?
The Mathematical Translation: $y = mx + b$
You likely remember the equation for a line from high school algebra: $y = mx + b$. In Data Science, we use slightly different notation, but the concept is identical:
$$y = \beta_0 + \beta_1x$$
This equation is not just math; it is a business narrative.
1. $y$ (Target): What we want to predict (Revenue). 2. $x$ (Feature): The lever we can pull (Marketing Spend). 3. $\beta_1$ (Coefficient/Slope): This is the most important number. It represents the impact. It tells us how much $y$ changes for every 1 unit increase in $x$. 4. $\beta_0$ (Intercept): This is the baseline. It represents the value of $y$ when $x$ is 0. (e.g., How much revenue would we make if we spent $0 on marketing? likely from word-of-mouth or existing contracts).
 A diagram illustrating the Linear Regression equation components on a chart. The 'Intercept' is highlighted where the line crosses the Y-axis. The 'Slope' is illustrated as a triangle stepping up along the line, labeled 'Rise over Run' or 'Change in Y divided by Change in X'. 

A diagram illustrating the Linear Regression equation components on a chart. The 'Intercept' is highlighted where the line crosses the Y-axis. The 'Slope' is illustrated as a triangle stepping up along the line, labeled 'Rise over Run' or 'Change in Y divided by Change in X'.
How the Machine "Learns": Minimizing Error
When we ask Scikit-Learn to "fit" a model, it uses an algorithm called Ordinary Least Squares (OLS).
Since real-world data is messy, no straight line will pass through every single data point perfectly. There will always be a gap between the actual data point and the predicted point on the line. This gap is called the Residual (or Error).
* The Goal: Find the specific line (slope and intercept) that makes the total sum of these squared errors as small as possible.
 A regression line cutting through scattered data points. Vertical lines are drawn connecting each data point to the regression line. These vertical lines are labeled 'Residuals' or 'Errors'. 

A regression line cutting through scattered data points. Vertical lines are drawn connecting each data point to the regression line. These vertical lines are labeled 'Residuals' or 'Errors'.
If the line is too steep, the errors get large. If the line is too flat, the errors get large. The "Best Fit Line" sits right in the "Goldilocks zone" where error is minimized.
Implementation in Python
Let’s simulate a scenario. You are the Data Scientist for an e-commerce company. Your CMO (Chief Marketing Officer) gives you data on ad spend and revenue for the last 10 months and asks: "If I spend $4,000 next month, exactly how much revenue should I expect?"
Here is how we solve this using Scikit-Learn.
1. Prepare the Data First, we import our libraries and create the dataset.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Sample Data: Marketing Spend (X) and Revenue (y)
data = {
    'marketing_spend': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500],
    'revenue': [12000, 18000, 23000, 29000, 34000, 40000, 47000, 53000, 58000, 64000]
}


df = pd.read_csv('marketing_data.csv') # Assuming we loaded this from a file


# SCALABILITY TIP: Scikit-Learn expects 'X' (features) to be a 2D array (a table), 
# and 'y' (target) to be a 1D array (a column).
```

X = df[['marketing_spend']]  # Double brackets make it a DataFrame (2D)
y = df['revenue']            # Single bracket makes it a Series (1D)
2. Instantiate and Fit We now initialize the algorithm and train it on our data.
```python
# 1. Instantiate the model
model = LinearRegression()


# 2. Fit the model (This is where OLS calculates the best line)
model.fit(X, y)
```

At this exact moment, Python has calculated the optimal $\beta_0$ and $\beta_1$. The "learning" is complete.
3. Extracting Insights Before we predict, we must interpret what the model learned. This is crucial for explaining the results to stakeholders.
```python
intercept = model.intercept_
coefficient = model.coef_[0]


print(f"Intercept (Baseline): ${intercept:.2f}")
print(f"Coefficient (Marketing Impact): {coefficient:.2f}")
```

**Output:**

```

```text
Intercept (Baseline): $444.44
```

Coefficient (Marketing Impact): 11.56
The Business Interpretation: The Baseline: Even if we turn off all marketing ads ($0 spend), our model estimates we would still make roughly $444 in revenue. The Impact: For every $1 dollar we add to the marketing budget, Revenue increases by $11.56. This is a powerful ROI metric to hand to your boss.
4. Making Predictions Finally, we answer the CMO's question: "What happens if we spend $4,000?"
```python
# Predict revenue for a spend of $4000
# Note: We must pass the input as a 2D array, hence the double brackets [[ ]]
new_spend = [[4000]]
predicted_revenue = model.predict(new_spend)


print(f"Projected Revenue for $4,000 spend: ${predicted_revenue[0]:,.2f}")
```

**Output:**

```

```text
```

Projected Revenue for $4,000 spend: $46,666.67
Visualizing the Model To verify our work, we should visualize the original data against our new regression line.
```python
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Marketing Spend')
plt.ylabel('Revenue')
plt.title('Simple Linear Regression: Spend vs Revenue')
plt.legend()
plt.show()
 The resulting plot from the code above. Blue dots represent the original data points. A bold red line cuts diagonally through the points, demonstrating a very close fit to the data. 

```

The resulting plot from the code above. Blue dots represent the original data points. A bold red line cuts diagonally through the points, demonstrating a very close fit to the data.
Summary We have successfully built a Simple Linear Regression model. We moved from historical data to a mathematical equation. We used OLS to minimize the error of that equation. We interpreted the coefficient* to understand the "exchange rate" between marketing dollars and revenue.
However, the real world is rarely driven by just one variable. Revenue isn't just driven by marketing; it's also driven by seasonality, competitor prices, and economic conditions. To handle that, we need to expand our toolkit to Multiple Linear Regression, which we will cover in the next section.
Feature Engineering: Selecting the Right Predictors
In the previous section, we built a "Crystal Ball" using Simple Linear Regression. We took a single input (Marketing Budget) and used it to forecast a single output (Sales).
While this was a fantastic first step, your intuition as a business professional probably signaled a limitation. In the real world, outcomes are rarely driven by a single factor.
If you are trying to predict the price of a house, you don't just look at the square footage. You also look at the number of bedrooms, the quality of the school district, the age of the roof, and the distance to the nearest highway.
To build models that reflect the complexity of reality, we must graduate from Simple Linear Regression to Multiple Linear Regression.
Instead of the formula looking like this: $$y = mx + b$$
It now looks like this: $$y = m_1x_1 + m_2x_2 + m_3x_3 + ... + b$$
Where $x_1$, $x_2$, and $x_3$ are different features (predictors) in your dataset.
However, adding more data brings a new challenge. Just because you have the data doesn't mean you should use it. This section focuses on Feature Selection: the art and science of choosing the right inputs to prevent your model from becoming confused, slow, or inaccurate.
The "Kitchen Sink" Trap
A common mistake for those transitioning into Data Science is the "Kitchen Sink" approach: throwing every available column of data into the model hoping it finds a pattern.
Imagine you are hiring a Sales Manager. You have a stack of resumes. To predict who will be the best hire, you look at: 1. Years of experience. 2. Past sales figures. 3. Industry contacts.
But would you also look at their shoe size? Or the day of the week they were born? Or their favorite ice cream flavor?
Obviously not. Those variables are noise. If you force a mathematical model to find a relationship between "Shoe Size" and "Sales Performance," it might accidentally find a coincidental pattern in your historical data. When you try to use that model on a new candidate, the prediction will fail because the relationship wasn't real.
 A split illustration. On the left, a funnel labeled "All Data" pouring into a machine, resulting in a graph with messy, erratic lines labeled "Overfitting/Noise". On the right, a filter labeled "Feature Selection" blocking irrelevant data (like "Shoe Size") while letting relevant data (like "Ad Spend") through to the machine, resulting in a clean, straight trend line. 

A split illustration. On the left, a funnel labeled "All Data" pouring into a machine, resulting in a graph with messy, erratic lines labeled "Overfitting/Noise". On the right, a filter labeled "Feature Selection" blocking irrelevant data (like "Shoe Size") while letting relevant data (like "Ad Spend") through to the machine, resulting in a clean, straight trend line.
The Two Golden Rules of Feature Selection
When selecting features for Linear Regression, we are generally looking for two things:
1. High Correlation with the Target: The feature should move in sync with what we are trying to predict. (e.g., As House Size goes up, Price goes up). 2. Low Correlation with Other Features: The features should be independent of one another.
We already understand Rule #1. Rule #2, however, is where many data scientists stumble. It introduces a concept called Multicollinearity.
Understanding Multicollinearity
Multicollinearity occurs when two or more predictors in your model are highly correlated with each other. They are essentially providing the model with the same information.
Imagine we are trying to predict the total revenue of a lemonade stand. Feature A: Number of cups sold. Feature B: Amount of revenue from cups sold.
If we include both A and B in the model to predict Total Revenue, the model gets confused. It doesn't know which variable to assign the "credit" to. Mathematically, this makes the model unstable. The coefficients (the $m$ in our equation) can swing wildly, making the model difficult to interpret.
Business Analogy: It is like having two employees work on the exact same task, but not telling them about each other. They might both do the work, or they might get in each other's way, and you end up paying double the salary for the same output.
Practical Workflow: The Correlation Matrix
How do we find the right features and avoid multicollinearity? We use a Correlation Matrix.
Let's look at a dataset for a hypothetical E-commerce company. We want to predict Yearly Amount Spent (Target). Our available features are: `Avg. Session Length`: Average time a user stays on the site. Time on App: Average time spent on the mobile app. `Time on Website`: Average time spent on the desktop site. Length of Membership: How many years they have been a customer.
Here is how we visualize these relationships using Python and the library seaborn.
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
customers = pd.read_csv('Ecommerce_Customers.csv')


# Calculate the correlation matrix
# .corr() computes the pairwise correlation of columns
correlation_matrix = customers.corr()


# Visualize with a Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of E-Commerce Features')
plt.show()
 A heatmap generated by Python. The diagonal shows dark red squares (1.0 correlation). The "Yearly Amount Spent" row shows varying shades. "Length of Membership" has a dark red square (0.8) indicating high correlation. "Time on Website" is a light blue square (0.02) indicating almost no correlation. "Time on App" is a medium red (0.5). 

```

A heatmap generated by Python. The diagonal shows dark red squares (1.0 correlation). The "Yearly Amount Spent" row shows varying shades. "Length of Membership" has a dark red square (0.8) indicating high correlation. "Time on Website" is a light blue square (0.02) indicating almost no correlation. "Time on App" is a medium red (0.5).
Interpreting the Heatmap
When you run this code, you will generate a grid of colored squares. Here is how to read it to select your features:
1. Look at the Target Row/Column: Find the row labeled Yearly Amount Spent. Look for colors indicating strong correlation (values close to 1.0 or -1.0). Observation: In our hypothetical plot, `Length of Membership` has a correlation of 0.8. This is a fantastic predictor. `Time on App` is 0.5. Good. `Time on Website` is -0.02*. This is noise; it has no relationship with spending. We should likely drop Time on Website.
2. Check for Multicollinearity: Look at the intersection of your features. Observation:* Check the intersection of Time on App and Length of Membership. If this value was very high (e.g., 0.9), we would have a problem (Multicollinearity). We would need to delete one of them. In this case, let's assume they are not correlated.
Implementing Multiple Linear Regression
Once we have selected our features (let's say we chose Length of Membership and Time on App), we simply update our X variable.
In Simple Linear Regression, X was a single column (a Pandas Series or 1D array). In Multiple Linear Regression, X is a DataFrame (a matrix).
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 1. Select Features (Predictors) and Target
# We drop 'Time on Website' because of low correlation seen in the heatmap
features = ['Length of Membership', 'Time on App']
target = 'Yearly Amount Spent'


X = customers[features]
y = customers[target]


# 2. Split the data (Standard Practice)
```

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 3. Instantiate the model
lm = LinearRegression()


# 4. Fit the model
# Scikit-Learn handles multiple features automatically!
lm.fit(X_train, y_train)


# 5. Inspect the Coefficients
print(f"Intercept: {lm.intercept_}")
print(f"Coefficients: {lm.coef_}")
Output Interpretation: The lm.coef_ will now print an array with two numbers, corresponding to the order of features you provided (Length of Membership, then Time on App).
If the output is [63.5, 38.2], the equation is:
$$Spending = 63.5 \times (\text{Membership Years}) + 38.2 \times (\text{App Time}) + \text{Intercept}$$
The Business Translation: "For every one year increase in Membership, spending increases by $63.50, holding all other factors constant." "For every one hour increase in App Time, spending increases by $38.20, holding all other factors constant."
Note the phrase "holding all other factors constant." This is the power of Multiple Linear Regression. It isolates the impact of one specific variable while accounting for the noise and influence of the others.
Summary We have moved from the simple world of single-variable prediction to the complex reality of multi-variable environments. By using Feature Selection, we ensure our model focuses on the signal and ignores the noise. By checking for Multicollinearity, we ensure our predictors aren't fighting each other for credit.
But we still have a lingering question. We built a model, and it produced an equation. But how do we know if the model is actually good? How accurate is it? In the next section, we will learn the critical metrics for Model Evaluation.
Case Study: Predicting Real Estate Prices
We have arrived at the convergence of theory and practice. In the previous sections, we learned the mechanics of Simple Linear Regression (one input, one output) and discussed Feature Engineering (the art of selecting meaningful inputs).
Now, we will combine these concepts to solve a classic, real-world business problem: Automated Valuation Models (AVMs). If you have ever used Zillow or Redfin to check the estimated value of a home, you have interacted with the technology we are about to build.
As a career-transitioning Data Scientist, you must move beyond just fitting a line to a scatter plot. You must now manage the entire modeling lifecycle: loading data, selecting multiple features, splitting data to prevent cheating (overfitting), training the model, and—crucially—explaining the results to a business stakeholder.
The Business Problem Imagine you have been hired by a Real Estate Investment Trust (REIT). They want to automate the bidding process for houses. Their current manual process takes too long, causing them to lose deals. They have provided you with a historical dataset of 1,000 homes sold in the last year, including details like square footage, number of bedrooms, age of the home, and the final selling price.
Your task: Build a model that predicts the SalePrice based on the property's characteristics.
Step 1: Loading and Inspecting the Data First, we load our libraries. We will use pandas for data handling and scikit-learn for the heavy lifting.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Load the dataset
# For this case study, we assume a cleaned CSV file named 'housing_data.csv'
df = pd.read_csv('housing_data.csv')


# Inspect the first few rows to understand our ingredients
print(df.head())
Output (Simulated):
```

```text
```

Square_Feet  Bedrooms  Bathrooms  Year_Built  Neighborhood  SalePrice
0         2100         4        2.5        1998       Suburb_A     320000
1         1600         3        2.0        1985       Suburb_B     250000
2         2800         4        3.0        2010       Suburb_A     410000
3         1200         2        1.0        1960       Urban_C      180000
4         1900         3        2.5        2005       Suburb_B     295000
Step 2: Feature Selection and Correlation In the previous section on Feature Engineering, we discussed that not all data points are useful predictors. A "Case Study ID" or "Homeowner Name" has no bearing on the market value of a house.
We need to select features that have a statistical relationship with SalePrice. A correlation matrix is our best tool here.
 A heatmap visualization generated by Seaborn. The X and Y axes list variables: Square_Feet, Bedrooms, Bathrooms, Year_Built, and SalePrice. The intersection of Square_Feet and SalePrice is colored dark red with a coefficient of 0.85, indicating strong positive correlation. The intersection of Bedrooms and SalePrice is moderately red (0.55). 

A heatmap visualization generated by Seaborn. The X and Y axes list variables: Square_Feet, Bedrooms, Bathrooms, Year_Built, and SalePrice. The intersection of Square_Feet and SalePrice is colored dark red with a coefficient of 0.85, indicating strong positive correlation. The intersection of Bedrooms and SalePrice is moderately red (0.55).
Based on our EDA (Exploratory Data Analysis), we observe that Square_Feet has the strongest relationship with price, but Year_Built and Bathrooms are also significant. We will define our Feature Matrix (X) and Target Vector (y).
Note: In this specific case study, we will focus on numerical features for Multiple Linear Regression. Handling categorical text data (like 'Neighborhood') requires a technique called One-Hot Encoding, which we will cover in Chapter 8.
```python
# Define our features (Inputs)
# We use double brackets [[]] to create a DataFrame, not a Series
features = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Year_Built']
X = df[features]


# Define our target (Output)
y = df['SalePrice']
```

Step 3: The Train-Test Split This is the most critical conceptual leap from "Statistics" to "Machine Learning."
If we show our model all the data during training, it will memorize the answers. It's like giving a student the answer key to the exam before they take it. They will get a 100% score, but they won't have learned how to solve the problems.
To assess if our model works on new, unseen houses, we split our data into two sets: 1. Training Set (80%): Used to teach the model. 2. Testing Set (20%): Locked away in a "vault." We only use this to grade the model at the very end.
 A diagram illustrating the Train-Test Split. A large rectangle represents the full dataset. It is sliced vertically. The left side (80%) is blue and labeled "Training Data (Model learns from this)". The right side (20%) is orange and labeled "Testing Data (Used for evaluation)". Arrows verify that the Model never sees the orange data until the prediction phase. 

A diagram illustrating the Train-Test Split. A large rectangle represents the full dataset. It is sliced vertically. The left side (80%) is blue and labeled "Training Data (Model learns from this)". The right side (20%) is orange and labeled "Testing Data (Used for evaluation)". Arrows verify that the Model never sees the orange data until the prediction phase.
```python
# Split the data
# random_state=42 ensures we get the same split every time we run the code (reproducibility)
```

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
Step 4: Instantiating and Fitting the Model Now we follow the Scikit-Learn workflow introduced earlier. Notice that the code for Multiple Linear Regression is identical to Simple Linear Regression. The algorithm handles the math of balancing multiple variables (coefficients) automatically.
```python
# 1. Instantiate the model
model = LinearRegression()


# 2. Fit the model (Learn the patterns)
# IMPORTANT: We fit ONLY on the training data
model.fit(X_train, y_train)


print("Model training complete.")
```

At this exact moment, the model object has calculated the "Line of Best Fit"—or rather, the "Hyperplane of Best Fit" since we are working in multiple dimensions. It has learned how much weight to give Square_Feet versus Bedrooms.
Step 5: Evaluating Performance How accurate is our crystal ball? To find out, we ask the model to predict the prices for the homes in our Testing Set (X_test). We then compare those predictions to the actual prices (y_test) which we hid from the model.
```python
# 3. Predict
predictions = model.predict(X_test)


# 4. Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared Score: {r2:.2f}")
Output (Simulated):
```

```text
```

Mean Absolute Error (MAE): $18,450.00
R-squared Score: 0.82
Interpreting the Metrics for Business Leaders As a Data Scientist, you cannot simply email the CEO saying "The R-squared is 0.82." You must translate this.
* The Translation: "Our model explains 82% of the variation in housing prices. On average, our automated price estimates are within \$18,450 of the actual selling price."
* The Decision: The business must decide if an error margin of ~\$18k is acceptable. If they are flipping high-end luxury homes, this is excellent accuracy. If they are buying \$50k fixer-uppers, this margin of error might be too high.
Step 6: Visualizing the Results Numbers are abstract; visuals are persuasive. A common way to diagnose a regression model is plotting Actual vs. Predicted values.
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions)


# Draw a red line representing perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)


plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Housing Prices')
plt.show()
 A scatter plot showing 'Actual Prices' on the X-axis and 'Predicted Prices' on the Y-axis. The dots cluster tightly around a red diagonal line running from bottom-left to top-right. This indicates high accuracy. A few outlier dots are far from the line, representing homes where the model predicted poorly. 

```

A scatter plot showing 'Actual Prices' on the X-axis and 'Predicted Prices' on the Y-axis. The dots cluster tightly around a red diagonal line running from bottom-left to top-right. This indicates high accuracy. A few outlier dots are far from the line, representing homes where the model predicted poorly.
If the dots fall exactly on the red line, the prediction is perfect. Dots significantly above or below the line represent errors. This visual helps you quickly identify if your model is failing on specific types of houses (e.g., maybe it consistently undervalues expensive mansions).
Step 7: The "Why" – Interpreting Coefficients Finally, we look inside the "Black Box." Because this is Linear Regression, we can see exactly how the model makes decisions by looking at the coefficients.
```python
coef_df = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print(coef_df)
Output (Simulated):
```

```text
Coefficient
Square_Feet       150.25
Bedrooms        -5000.00
Bathrooms       12000.00
Year_Built        800.00
```

Wait, why is the coefficient for Bedrooms negative? This is a common analytical trap. It suggests that, holding all other variables constant, adding a bedroom reduces the price by \$5,000.
Mathematically, this happens because Square_Feet is already in the model. If you have two houses that are both 2,000 sq ft, but one has 3 bedrooms and the other has 5 bedrooms, the 5-bedroom house must have extremely tiny rooms (chopped up space), which might be less desirable.
This reinforces the lesson from the Correlation vs. Causation section: model coefficients describe mathematical relationships, not necessarily physical laws.
Summary In this case study, you successfully: 1. Loaded and inspected raw data. 2. Selected features based on correlation. 3. Split data to validate performance on unseen records. 4. Trained a Multiple Linear Regression model. 5. Translated technical metrics (MAE) into business risk (Dollars).
You have moved from describing the past to predicting the future. However, you may have noticed that our model assumed the relationship between size and price is a straight line. What if the relationship is curved? What if location matters more than size? In the next chapter, we will explore how to handle these complexities using non-linear models.
