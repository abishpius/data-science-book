# Chapter 5: Exploratory Data Analysis and Visualization

The Grammar of Graphics with Matplotlib and Seaborn
If you have ever presented a business report, you know that a spreadsheet of numbers, no matter how well-cleaned, rarely tells a compelling story. In the previous section, we identified outliers mathematically—finding the "Bill Gates in the bar" scenario. But numbers alone can be abstract. To truly understand the shape of our data, the skew of our distributions, and the relationships between variables, we need to visualize them.
In Excel, creating a chart is a point-and-click adventure. You highlight a range of cells, click "Insert Chart," and then navigate menus to change colors or labels. It is intuitive, but it is also manual and difficult to reproduce perfectly next month.
In Python, visualization is done through code. This allows you to automate your reporting and handle millions of data points that would crash Excel. However, it requires a mental shift from "selecting and clicking" to "building in layers." This approach is often referred to as the Grammar of Graphics.
The Hierarchy of a Plot: Figure and Axes
The most popular library for visualization in Python is Matplotlib. It is the foundation upon which almost all other Python visualization tools are built.
When transitioning from Excel, the most confusing part of Matplotlib is understanding the "canvas." In Excel, a chart is an object floating on a sheet. In Matplotlib, we must explicitly define the hierarchy of the image.
Think of it like painting: 1. The Figure (`fig`): This is your blank canvas or the physical piece of paper. It holds everything. 2. The Axes (`ax`): This is not just the x and y-axis lines; it is the specific area on the canvas where the data is drawn. A single Figure can contain multiple Axes (like a comic book page with multiple panels).
 A diagram illustrating the Matplotlib hierarchy. The outer box is labeled 'Figure'. Inside, there is a smaller box labeled 'Axes'. Within the Axes, distinct elements are pointed out: the 'X-Axis', 'Y-Axis', 'Title', 'Tick Marks', and the 'Data Line' itself. 

A diagram illustrating the Matplotlib hierarchy. The outer box is labeled 'Figure'. Inside, there is a smaller box labeled 'Axes'. Within the Axes, distinct elements are pointed out: the 'X-Axis', 'Y-Axis', 'Title', 'Tick Marks', and the 'Data Line' itself.
Here is the standard pattern for creating a plot in Python using the subplots method:
```python
import matplotlib.pyplot as plt


# 1. Prepare the data (Simulating the cleaned data from previous sections)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
revenue = [10000, 12500, 11000, 16000, 15500]


# 2. Create the Figure and Axes objects
fig, ax = plt.subplots(figsize=(10, 6))


# 3. Plot data onto the Axes
ax.plot(months, revenue, marker='o', linestyle='-', color='blue')


# 4. Customize the Axes (Labels, Titles)
ax.set_title("Monthly Revenue Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")


# 5. Display the result
plt.show()
```

Notice the difference from Excel? We aren't clicking specific elements to change them; we are calling methods on the ax object (like ax.set_title). This makes your analysis reproducible. If the data changes next month, you re-run the script, and the chart updates instantly with the exact same formatting.
Seaborn: A High-Level Wrapper
While Matplotlib is powerful, it can be verbose. Creating a complex statistical plot might require dozens of lines of code. Enter Seaborn.
Seaborn is a library built on top of Matplotlib. If Matplotlib is the engine, Seaborn is the sleek dashboard. It is designed specifically for Data Science and integrates tightly with Pandas DataFrames.
Seaborn simplifies the process by: 1. Accepting a DataFrame directly. 2. Automatically applying aesthetically pleasing themes. 3. Calculating statistics (like confidence intervals) for you automatically.
Let’s look at the "Bill Gates" outlier problem from the previous section. In Excel, spotting outliers usually involves scanning rows. In Python, we use a Boxplot, which visually summarizes the distribution and highlights anomalies as dots beyond the "whiskers."
 An anatomical breakdown of a Boxplot. It labels the 'Median' (line in the box), 'Interquartile Range' (the box itself), 'Whiskers' (lines extending out), and 'Outliers' (individual dots floating beyond the whiskers). 

An anatomical breakdown of a Boxplot. It labels the 'Median' (line in the box), 'Interquartile Range' (the box itself), 'Whiskers' (lines extending out), and 'Outliers' (individual dots floating beyond the whiskers).
Here is how we visualize distributions and outliers using Seaborn:
```python
import seaborn as sns
import pandas as pd


# Simulating a dataset with an outlier (The "Bill Gates" scenario)
data = {
    'Department': ['Sales'] * 20,
    'Salary': [50000, 52000, 48000, 51000, 49500, 53000, 51000, 
               50500, 49000, 52500, 51500, 48500, 50000, 51000, 
               52000, 49500, 50500, 48000, 51000, 1000000] # The outlier
}
df = pd.DataFrame(data)


# Create the Figure and Axes
fig, ax = plt.subplots(figsize=(8, 4))


# Create a Boxplot using Seaborn
# Notice we pass the DataFrame (df) and the column names directly
sns.boxplot(data=df, x='Salary', ax=ax)


ax.set_title("Salary Distribution (Detecting Outliers)")
plt.show()
```

In the resulting chart, the outlier ($1,000,000) would appear as a lone dot far to the right, while the box would show where the majority of employees sit. This visualization provides an immediate "sanity check" on your data that a simple average calculation would hide.
Univariate vs. Bivariate Analysis
When performing Exploratory Data Analysis (EDA), you generally move through two stages using these tools:
1. Univariate Analysis: Looking at one variable at a time. Question: What is the distribution of transaction values? Tool: Histogram or Boxplot. Library:* sns.histplot() or sns.boxplot().
2. Bivariate Analysis: Looking at the relationship between two variables. Question: Does higher marketing spend lead to higher sales? Tool: Scatter plot or Line chart. Library:* sns.scatterplot() or sns.lineplot().
In Excel, creating a scatter plot with thousands of points often slows the application to a crawl. Python handles this effortlessly.
```python
# Bivariate Example: Marketing Spend vs Revenue
market_data = {
    'Marketing_Spend': [100, 200, 300, 400, 500, 1000],
    'Revenue': [120, 250, 310, 480, 520, 1100],
    'Region': ['North', 'North', 'South', 'South', 'East', 'East']
}
df_market = pd.DataFrame(market_data)


# Plotting with a third dimension (Color/Hue)
fig, ax = plt.subplots(figsize=(8, 6))


sns.scatterplot(
    data=df_market, 
    x='Marketing_Spend', 
    y='Revenue', 
    hue='Region',   # Colors dots based on Region
    s=100,          # Size of dots
    ax=ax
)


ax.set_title("Impact of Marketing on Revenue by Region")
plt.show()
```

Combining Powers: Matplotlib and Seaborn Together
Because Seaborn is built on Matplotlib, you can mix their commands. You use Seaborn to do the heavy lifting of drawing the complex statistical shape, and you use Matplotlib to tweak the formatting "around" the plot.
This is the workflow you will use most often as a Data Scientist: 1. Use Pandas to filter and aggregate the data (weighing the haystacks). 2. Use Matplotlib to set up the figure canvas (plt.subplots). 3. Use Seaborn to draw the visualization onto that canvas. 4. Use Matplotlib again to refine labels, limits, and titles for the final report.
By mastering this grammar of graphics, you move beyond simply storing data to communicating insights. In the next section, we will explore how to perform Feature Engineering—creating new metrics from existing data—to feed into these visualizations for deeper analysis.
Univariate Analysis: Understanding Distributions and Spread
In the previous section, we introduced the concept of the "Grammar of Graphics"—the idea that a plot is a mapping of data variables to aesthetic attributes like x-axis, y-axis, color, and shape. We learned how to import Matplotlib and Seaborn to set up our visualization canvas.
Now, we apply that grammar to the most fundamental step of Exploratory Data Analysis (EDA): Univariate Analysis.
"Univariate" simply means "one variable." Before you can understand how Marketing Spend impacts Total Revenue (multivariate analysis), you must deeply understand Marketing Spend and Total Revenue individually. In the business world, this is equivalent to doing a basic health check on your metrics before trying to diagnose complex strategy problems.
We need to answer two main questions for every column in our dataset: 1. Central Tendency: What is the "common" or "average" value? 2. Spread (Dispersion): How much does the data vary? Is it consistent, or is it volatile?
Analyzing Categorical Variables
For categorical data (like Region, Product Category, or Payment Method), "distribution" simply refers to frequency. We want to know how many times each category appears.
In Excel, you would typically handle this by creating a Pivot Table, dragging a category to "Rows" and the same category to "Values" to get a count. In Python, pandas handles this with a single method: .value_counts().
Let's create a sample dataset to demonstrate this.
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Creating a sample retail dataset
data = {
    'Transaction_ID': range(1, 11),
    'Region': ['North', 'North', 'South', 'East', 'North', 'West', 'South', 'North', 'East', 'West'],
    'Sales_Amount': [100, 120, 60, 200, 110, 90, 70, 400, 210, 95]
}
df = pd.DataFrame(data)


# Frequency count of Regions
region_counts = df['Region'].value_counts()
print(region_counts)
```

**Output:**

```

```text
North    4
South    2
East     2
West     2
```

Name: Region, dtype: int64
While the text output is precise, a visualization allows you to instantly spot imbalances in your data (e.g., if 90% of your sales are coming from the 'North' region). The standard tool for this is the Count Plot.
```python
# Visualizing the frequency of a categorical variable
plt.figure(figsize=(8, 5))
sns.countplot(x='Region', data=df, palette='viridis')
plt.title('Distribution of Transactions by Region')
plt.show()
```

This generates a bar chart displaying the frequency of each category. If you are coming from Excel, note that you did not need to aggregate the data first. Seaborn’s countplot calculates the counts automatically from the raw rows.
Analyzing Numerical Variables: The Histogram
When dealing with continuous numbers (like Sales_Amount, Age, or Temperature), we cannot simply count the unique values because almost every row might have a slightly different number (e.g., 100.01 vs 100.02).
Instead, we group these numbers into "bins" (intervals). This creates a Histogram.
Imagine you have a bucket for sales between \$0-\$50, another for \$51-\$100, and so on. You drop every transaction into the corresponding bucket and count how high the pile gets.
 A diagram showing the construction of a histogram. The left side shows a list of raw numbers. The right side shows these numbers being sorted into bins (0-10, 10-20, 20-30), forming bars that represent the frequency of data within those ranges. 

A diagram showing the construction of a histogram. The left side shows a list of raw numbers. The right side shows these numbers being sorted into bins (0-10, 10-20, 20-30), forming bars that represent the frequency of data within those ranges.
In Python, we visualize this using sns.histplot. We also often add a Kernel Density Estimate (KDE)—a smooth line that traces the shape of the distribution.
```python
# Visualizing the distribution of Sales Amount
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Sales_Amount', bins=5, kde=True)
plt.title('Distribution of Sales Amount')
plt.xlabel('Revenue ($)')
plt.ylabel('Frequency')
plt.show()
```

Interpreting the Shape: Skewness The shape of the histogram tells you a story about your business.
1. Normal Distribution (Bell Curve): The mean, median, and mode are roughly the same. Most sales are in the middle; very low and very high sales are rare. 2. Right Skewed (Positive Skew): The tail extends to the right. This is classic for financial data. Most transactions are small (the "hump" is on the left), but a few massive "whale" clients pull the tail to the right. 3. Left Skewed (Negative Skew): The tail extends to the left. This might happen in customer satisfaction scores (1-10), where most people give 8s or 9s, but a few angry customers give 1s and 2s.
 A comparison of three distribution shapes. 1. Normal Distribution (symmetrical bell curve). 2. Right Skewed Distribution (hump on left, long tail stretching right). 3. Left Skewed Distribution (hump on right, long tail stretching left). Annotations indicate where the Mean vs. Median sits in each skew. 

A comparison of three distribution shapes. 1. Normal Distribution (symmetrical bell curve). 2. Right Skewed Distribution (hump on left, long tail stretching right). 3. Left Skewed Distribution (hump on right, long tail stretching left). Annotations indicate where the Mean vs. Median sits in each skew.
Measures of Spread: Box Plots
While the histogram shows the shape, it can sometimes obscure spread and outliers, especially if the bin sizing isn't perfect. As we discussed in the "Outliers" section, averages can be deceiving.
To visualize the spread and robustness of the data, Data Scientists rely on the Box Plot (also known as the Box-and-Whisker plot).
If you haven't used these in Excel, they may look intimidating, but they are standardized summaries of five key numbers: 1. Minimum: The lowest value (excluding outliers). 2. Q1 (25th Percentile): 25% of the data is lower than this line. 3. Median (50th Percentile): The exact middle of the dataset. 4. Q3 (75th Percentile): 75% of the data is lower than this line. 5. Maximum: The highest value (excluding outliers).
The "Box" represents the middle 50% of your data (the Interquartile Range, or IQR). This is where the "normal" business happens.
 Anatomy of a Box Plot. A detailed diagram labeling the Box (IQR), the horizontal line inside the box (Median), the vertical lines extending out (Whiskers), and individual dots beyond the whiskers labeled as "Outliers". 

Anatomy of a Box Plot. A detailed diagram labeling the Box (IQR), the horizontal line inside the box (Median), the vertical lines extending out (Whiskers), and individual dots beyond the whiskers labeled as "Outliers".
Let's visualize our Sales Amount using a box plot.
```python
# A Box Plot to check for spread and outliers
plt.figure(figsize=(8, 3))
sns.boxplot(x=df['Sales_Amount'], color='lightblue')
plt.title('Box Plot of Sales Amount')
plt.show()
```

How to read this as a Business Analyst: The Line in the Box: This is your Median sales value. It is more robust than the Average. The Width of the Box: If the box is narrow, your customers spend very consistent amounts. If the box is wide, your customer spending behavior is volatile. The Dots:* Any dots outside the whiskers are outliers. In our dataset, the transaction of 400 likely appears as a dot on the far right. This signals you need to investigate that specific transaction—is it a bulk order? A data entry error?
Summary Statistics with .describe()
Visualizations are powerful, but sometimes you just need the raw numbers. In Excel, you might use the Analysis ToolPak to generate descriptive statistics. In Python, pandas provides the .describe() method.
This method changes behavior based on the data type.
For Numerical Data:
```python
print(df['Sales_Amount'].describe())
```

Output includes count, mean, std (standard deviation), min, 25%, 50%, 75%, and max.
For Categorical Data (Object):
```python
print(df['Region'].describe())
```

Output includes count, unique (how many categories), top (most frequent category), and freq (count of the top category).
Moving Forward
By performing Univariate Analysis, you have established a baseline. You know that your sales are Right Skewed, you know that the 'North' region is your most frequent market, and you've identified a few high-value outliers using a box plot.
However, analyzing variables in isolation can only take you so far. Business value is usually found in the relationships between variables. Does the 'North' region actually spend more per transaction than the 'South'? Does higher marketing spend actually correlate with higher sales?
In the next section, we will move to Bivariate Analysis, where we will explore correlations and relationships between two variables.
Bivariate Analysis: Visualizing Correlations and Trends
In the previous section, we focused on Univariate Analysis—examining variables in isolation. We learned to calculate the mean of our sales data, visualize the spread of customer ages using histograms, and identify the skew in our profit margins.
While Univariate Analysis helps us understand what our data looks like, it rarely tells us why things are happening. In a business context, knowing that the average daily revenue is $5,000 is useful, but knowing that Marketing Spend drives that revenue is actionable.
This brings us to Bivariate Analysis ("Bi" meaning two). Here, we move from taking portraits of individual variables to observing the conversations between them. We are looking for relationships, correlations, and trends.
Numerical vs. Numerical: The Scatter Plot
The most common relationship you will investigate in data science is between two continuous numerical variables. In Excel, you might highlight two columns and insert a scatter chart to see if they move together. In Python, we use the Grammar of Graphics to map one variable to the x-axis and another to the y-axis.
Let’s imagine we are analyzing an e-commerce dataset. We suspect a relationship between the number of Site_Visitors and Total_Sales.
 A scatter plot showing a strong positive correlation. The X-axis is labeled "Daily Site Visitors" and the Y-axis is labeled "Total Sales ($)". The points drift upward from left to right, indicating that as visitors increase, sales increase. 

A scatter plot showing a strong positive correlation. The X-axis is labeled "Daily Site Visitors" and the Y-axis is labeled "Total Sales ($)". The points drift upward from left to right, indicating that as visitors increase, sales increase.
Here is how we generate this view using Seaborn:
```python
import matplotlib.pyplot as plt
import seaborn as sns


# Assuming 'df' is our DataFrame containing e-commerce data
plt.figure(figsize=(10, 6))


# Plotting Visitors vs Sales
sns.scatterplot(data=df, x='Site_Visitors', y='Total_Sales', alpha=0.6)


plt.title('Relationship Between Site Traffic and Sales')
plt.xlabel('Daily Site Visitors')
plt.ylabel('Total Sales ($)')
plt.show()
```

Key concept: Notice the argument alpha=0.6. In large datasets with thousands of points, dots often overlap, creating a solid blob of ink. The alpha parameter controls transparency (0 is invisible, 1 is solid). By making points semi-transparent, overlapping areas become darker, revealing the density of the data.
Visualizing the Trendline Sometimes the relationship isn't immediately obvious, or you want to model the general direction. In Excel, you would right-click a data point and select "Add Trendline." In Seaborn, we switch from scatterplot to regplot (Regression Plot).
```python
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Site_Visitors', y='Total_Sales', line_kws={"color": "red"})
plt.title('Traffic vs Sales with Linear Trend')
plt.show()
```

This draws a scatter plot and automatically overlays a linear regression model (the line of best fit) with a shaded confidence interval. If the line slopes upward, the variables are positively correlated; if downward, negatively correlated.
Quantifying Relationships: Correlation Matrices
Visuals are powerful, but subjective. A scatter plot might look like a "strong" relationship to one stakeholder and "weak" to another. To make this objective, we use the Correlation Coefficient (specifically, Pearson correlation).
This value ranges from -1 to 1: 1: Perfect positive correlation (As X goes up, Y goes up). 0: No correlation (Random noise). -1:* Perfect negative correlation (As X goes up, Y goes down—like "Price" vs. "Demand").
In pandas, calculating this is incredibly efficient compared to Excel's CORREL formulas:
```python
# Calculate correlations for all numerical columns
correlation_matrix = df.corr()
print(correlation_matrix)
```

However, a table of raw numbers is hard to read. The best way to present correlation in a professional setting is using a Heatmap. A heatmap replaces numbers with colors, allowing you to spot "hot spots" (strong relationships) instantly.
 A correlation heatmap using a "coolwarm" color palette. The diagonal squares are dark red (correlation of 1.0). A square at the intersection of "Marketing_Spend" and "Revenue" is distinctively red, indicating high correlation. A square between "Price" and "Purchase_Frequency" is blue, indicating negative correlation. 

A correlation heatmap using a "coolwarm" color palette. The diagonal squares are dark red (correlation of 1.0). A square at the intersection of "Marketing_Spend" and "Revenue" is distinctively red, indicating high correlation. A square between "Price" and "Purchase_Frequency" is blue, indicating negative correlation.
```python
plt.figure(figsize=(10, 8))


# Create the heatmap
sns.heatmap(correlation_matrix, 
            annot=True,        # Write the data value in each cell
            cmap='coolwarm',   # Color scheme (Blue for negative, Red for positive)
            fmt=".2f")         # Format to 2 decimal places


plt.title('Correlation Matrix of Business Metrics')
plt.show()
```

Numerical vs. Categorical: Comparing Groups
Bivariate analysis isn't limited to just numbers. Often, the most valuable business questions involve categories. For example: Do sales differ by Region? or Is the Shipping Cost higher for 'Express' vs 'Standard'?
In the previous section regarding Univariate Analysis, we used the Boxplot to see the distribution of one variable. By adding a second variable (the category) to the x-axis, we can perform side-by-side comparisons.
Imagine analyzing salary data across different departments.
```python
plt.figure(figsize=(12, 6))


# x is the Category, y is the Numerical value
sns.boxplot(data=df, x='Department', y='Salary')


plt.title('Salary Distribution by Department')
plt.show()
 A set of side-by-side boxplots. The X-axis lists departments: Sales, IT, HR, Marketing. The Y-axis represents Salary. The "IT" boxplot is positioned higher on the Y-axis than "HR", and has a wider "box," indicating higher median pay but also higher variance. Outliers (dots) are visible above the whiskers. 

```

A set of side-by-side boxplots. The X-axis lists departments: Sales, IT, HR, Marketing. The Y-axis represents Salary. The "IT" boxplot is positioned higher on the Y-axis than "HR", and has a wider "box," indicating higher median pay but also higher variance. Outliers (dots) are visible above the whiskers.
Interpreting this plot: 1. The Line in the Middle: Comparing the median lines tells you which department pays more on average (robust to outliers). 2. The Height of the Box: A tall box means high variability—some people earn very little, some earn a lot. A short box means salaries are consistent. 3. The Whiskers: These show the range of "normal" salaries. 4. The Dots: These are outliers. If the "Sales" department has many dots above the top whisker, it indicates heavy commissions or distinct high-performers.
Categorical vs. Categorical: Cross-Tabulation
Finally, you may need to analyze the relationship between two categorical variables. Is there a relationship between 'Churn Status' (Yes/No) and 'Contract Type' (Month-to-Month/Yearly)?*
In Excel, you would solve this with a Pivot Table using "Count." In Python, we can visualize this using a Countplot with a hue argument. The hue parameter adds a second categorical dimension by color-coding the bars.
```python
plt.figure(figsize=(10, 6))


# Comparing Contract Type vs Churn
sns.countplot(data=df, x='Contract_Type', hue='Churn')


plt.title('Customer Churn by Contract Type')
plt.ylabel('Number of Customers')
plt.show()
 A grouped bar chart. The X-axis has three categories: Month-to-Month, One Year, Two Year. Each category has two bars side-by-side: Blue for "Churn: No" and Orange for "Churn: Yes". The "Month-to-Month" group has a very high Orange bar compared to the "Two Year" group, visually proving that short-term contracts lead to higher churn. 

```

A grouped bar chart. The X-axis has three categories: Month-to-Month, One Year, Two Year. Each category has two bars side-by-side: Blue for "Churn: No" and Orange for "Churn: Yes". The "Month-to-Month" group has a very high Orange bar compared to the "Two Year" group, visually proving that short-term contracts lead to higher churn.
This visualization instantly validates business hypotheses. If the "Month-to-Month" bars show a high proportion of churn compared to "Two Year" contracts, you have data-driven evidence to suggest incentivizing long-term contracts.
Summary: The Analyst's Toolkit
You have now expanded your toolkit from understanding single variables to understanding relationships.
* Use Scatter Plots (`scatterplot`) to see how two numbers move together.
* Use Correlation Heatmaps (`heatmap`) to scan for drivers and relationships across the whole dataset.
* Use Boxplots (`boxplot`) to compare numerical distributions across different categories.
* Use Countplots (`countplot` with `hue`) to compare frequencies between two categories.
However, the real world is rarely defined by just two variables. Sales aren't just driven by Marketing Spend; they are driven by Spend, Seasonality, Competitor Prices, and Inventory Levels simultaneously. In the next section, we will introduce Multivariate Analysis, where we learn to visualize three or more dimensions of data at once.
Case Study: Diagnosing Sales Performance Factors
In the previous sections, we have built a robust toolkit. We know how to clean messy text, handle data types, identify outliers, and apply the Grammar of Graphics to visualize single variables and relationships.
However, in the professional world, your boss will rarely ask you to "perform a bivariate analysis on column X and Y." Instead, they will ask a business question: "Why did our revenue drop in Q3?" or "Are our marketing promotions actually working?"
This section is where the rubber meets the road. We will combine all the techniques we have learned into a comprehensive workflow—a Case Study. We will step into the role of a Data Scientist at a mid-sized electronics retailer, "TechGear," to diagnose specific factors driving sales performance.
The Scenario: TechGear's Profitability Puzzle
TechGear’s executive team is concerned. While overall revenue is high, profit margins in certain segments seem to be eroding. They have provided you with a dataset of recent transactions and asked you to investigate. They have two specific hypotheses they want you to test: 1. Are specific regions underperforming? 2. Is our discounting strategy hurting our profitability?
Step 1: Setup and Data Cleaning
Before we can visualize anything, we must ensure our foundation is solid. As we learned in the "String Manipulation" section, categorical data is often messy.
Let's import our libraries and simulate the data loading process.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting the visual style for our charts
sns.set_theme(style="whitegrid")


# Loading the dataset (simulated for this example)
df = pd.read_csv('techgear_sales_data.csv')


# Inspecting the raw data
print(df.head())
```

Imagine the output reveals the Region column contains inconsistent entries like " North", "north", and "North ". If we plot this immediately, Python will treat these as three different regions. We need to standardize this text.
```python
# 1. String Manipulation: Standardizing Region names
# Strip whitespace and convert to title case
df['Region'] = df['Region'].str.strip().str.title()


# 2. Data Typing: Ensure Date is a datetime object
df['Date'] = pd.to_datetime(df['Date'])


print(df['Region'].unique())
# Output: ['North', 'South', 'East', 'West'] -> Clean!
 A flow diagram illustrating the data cleaning pipeline. On the left, a cylinder labeled "Raw Data" feeds into a funnel. Inside the funnel, icons represent "String Stripping," "Capitalization," and "Type Casting." On the right, a neat table emerges labeled "Analysis Ready Data." 

```

A flow diagram illustrating the data cleaning pipeline. On the left, a cylinder labeled "Raw Data" feeds into a funnel. Inside the funnel, icons represent "String Stripping," "Capitalization," and "Type Casting." On the right, a neat table emerges labeled "Analysis Ready Data."
Step 2: Univariate Analysis (The "Sanity Check")
Before looking for relationships, we must understand the distribution of our key metric: Sales Amount. As discussed in the section on Outliers, financial data is rarely normally distributed; it often follows a "power law" where a few large deals skew the average.
```python
# Visualizing the distribution of Sales Amount
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Sales_Amount', kde=True, bins=30)
plt.title('Distribution of Transaction Values')
plt.xlabel('Sales Amount ($)')
plt.show()
 A histogram with a kernel density estimate (KDE) line overlay. The x-axis is "Sales Amount ($)" and the y-axis is "Count". The distribution is "right-skewed," meaning there is a tall peak on the left (many small transactions) and a long tail extending to the right (a few very expensive transactions). 

```

A histogram with a kernel density estimate (KDE) line overlay. The x-axis is "Sales Amount ($)" and the y-axis is "Count". The distribution is "right-skewed," meaning there is a tall peak on the left (many small transactions) and a long tail extending to the right (a few very expensive transactions).
Interpretation: If the histogram showed a massive spike at \$0, we might have a data quality issue (e.g., failed transactions recorded as zero). If it is heavily right-skewed (as shown in the figure), we know that using the "mean" might be misleading. We should perhaps use the "median" for our business reporting.
Step 3: Bivariate Analysis (Testing Hypothesis 1)
The management's first question was: Are specific regions underperforming?
To answer this, we need to compare a categorical variable (Region) against a numerical variable (Sales_Amount). As we learned in the Grammar of Graphics, a Boxplot is the ideal geometric object for this comparison because it shows the median, the spread, and the outliers simultaneously.
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Region', y='Sales_Amount', palette='Set2')
plt.title('Sales Performance by Region')
plt.show()
 A boxplot displaying four regions (North, South, East, West) on the x-axis. The y-axis represents "Sales_Amount". The "South" box is significantly shorter and lower on the y-axis than the others, indicating lower median sales and less variability. The "North" box has several dots above the top whisker, indicating high-value outliers. 

```

A boxplot displaying four regions (North, South, East, West) on the x-axis. The y-axis represents "Sales_Amount". The "South" box is significantly shorter and lower on the y-axis than the others, indicating lower median sales and less variability. The "North" box has several dots above the top whisker, indicating high-value outliers.
The Insight: The plot immediately highlights that the South region has a lower median transaction value compared to the North. However, the North relies heavily on outliers (massive, rare deals) to keep its numbers up. This is a nuance that a simple Excel pivot table averaging the numbers might miss.
Step 4: Multivariate Analysis (Testing Hypothesis 2)
The second question is trickier: Is our discounting strategy hurting profitability?
This requires us to look at three variables at once: 1. Discount % (Independent Variable) 2. Profit (Dependent Variable) 3. Product Category (Grouping Variable - to see if this applies to all products)
We will use a Scatter Plot, which is excellent for showing the correlation between two continuous variables. We will add a third dimension using hue (color) to separate product categories.
```python
plt.figure(figsize=(12, 8))


# Scatter plot correlating Discount to Profit
sns.scatterplot(
    data=df, 
    x='Discount_Pct', 
    y='Profit', 
    hue='Category', 
    alpha=0.7 # Transparency helps visualize overlapping points
)


# Adding a horizontal line at 0 to mark the break-even point
plt.axhline(0, color='red', linestyle='--', linewidth=1)


plt.title('Impact of Discounting on Profitability by Category')
plt.xlabel('Discount Percentage (0.1 = 10%)')
plt.ylabel('Profit ($)')
plt.show()
 A scatter plot with "Discount Percentage" on the x-axis and "Profit" on the y-axis. A red dashed horizontal line runs across y=0. Points are colored by category (Electronics, Clothing, Home). The trend shows that as the Discount Percentage moves to the right (increases), the Profit points drift downward. Specifically, the "Electronics" points drop below the red line (negative profit) once the discount exceeds 20%. 

```

A scatter plot with "Discount Percentage" on the x-axis and "Profit" on the y-axis. A red dashed horizontal line runs across y=0. Points are colored by category (Electronics, Clothing, Home). The trend shows that as the Discount Percentage moves to the right (increases), the Profit points drift downward. Specifically, the "Electronics" points drop below the red line (negative profit) once the discount exceeds 20%.
The Insight: The visual analysis reveals a critical threshold. While discounts generally lower profit (an expected negative correlation), the category Electronics actually becomes unprofitable (drops below the red line) when discounts exceed 20%.
This suggests that the sales team is aggressive with discounts to close deals, but for low-margin electronics, they are essentially paying customers to take the inventory.
Summary of the Workflow
Notice how we didn't just "make charts." We followed a diagnostic path:
1. Data Type/String Cleaning: We ensured our Region labels were unified so our groups were accurate. 2. Univariate Analysis: We established a baseline for what a "normal" sale looks like. 3. Bivariate Analysis: We identified that the South region is underperforming in transaction value. 4. Multivariate Analysis: We diagnosed that aggressive discounting in Electronics is destroying profit margins.
This is the essence of Exploratory Data Analysis. It is not about generating code; it is about interrogating data to find the narrative hidden within the numbers. In the next chapter, we will move from analyzing historical data to predicting future data using Machine Learning.
