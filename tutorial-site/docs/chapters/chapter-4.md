# Chapter 4: Data Cleaning and Preparation

Handling Missing Data: Imputation vs. Deletion Strategies
Data in the real world is rarely pristine. In our previous sections, we ingested CSVs and performed aggregations assuming that every row contained perfect, complete information. We assumed every customer had an age, every transaction had a dollar amount, and every product had a category.
In reality, you will encounter datasets that look like Swiss cheese. A survey respondent skipped a question; a sensor went offline for an hour; a legacy database export corrupted a specific text field.
In Excel, a blank cell is visually obvious. You might manually scan the sheet, filter for "Blanks," or use an IF(ISBLANK(), ...) formula. In Pandas, missing data is represented as NaN (Not a Number) or None. If you attempt to calculate the sum or average of a column containing NaN values without handling them, your analysis may either crash or, worse, silently produce skewed results.
This section focuses on the two primary strategies for handling these gaps: Deletion (removing the data) and Imputation (guessing the data).
Identifying the Gaps
Before you fix the problem, you must quantify it. You cannot simply "look" at a DataFrame with 50,000 rows to find blank cells.
Let's imagine a DataFrame named crm_data containing customer leads.
```python
import pandas as pd
import numpy as np


# Sample data with intentional gaps
data = {
    'Company': ['Acme Corp', 'Globex', 'Soylent Corp', 'Initech', 'Umbrella Corp'],
    'Revenue': [1000000, np.nan, 500000, 75000, np.nan],
    'Employee_Count': [50, 1200, np.nan, 10, 500],
    'Sector': ['Tech', 'Logistics', 'Food', 'Tech', np.nan]
}


crm_data = pd.DataFrame(data)
```

If you print this DataFrame, you will see NaN where data is missing. To get a high-level summary of your "data hygiene," use the .info() method or chain .isnull().sum().
```python
# Check for missing values per column
print(crm_data.isnull().sum())
```

**Output:**

```

```text
Company           0
Revenue           2
Employee_Count    1
Sector            1
dtype: int64
 A visual representation of a DataFrame heatmap using the 'seaborn' library. The heatmap shows yellow bars representing valid data and black gaps representing missing (NaN) values, illustrating how data gaps can be distributed randomly across rows and columns. 

```

A visual representation of a DataFrame heatmap using the 'seaborn' library. The heatmap shows yellow bars representing valid data and black gaps representing missing (NaN) values, illustrating how data gaps can be distributed randomly across rows and columns.
Once you know where the holes are, you have to make a business decision: do you cut the rot out, or do you try to repair it?
Strategy 1: Deletion (The "Nuclear" Option)
Deletion is the simplest approach. If a row is incomplete, you remove it. In statistical terms, this is "Listwise Deletion."
In Pandas, we use the .dropna() method.
Dropping Rows If you are analyzing "Revenue per Company," a row with no Revenue data is useless to you.
```python
# Drop any row that contains at least one missing value
clean_rows = crm_data.dropna()
```

However, be careful. If you have a dataset with 10 columns, and a row is missing data in only one unimportant column (e.g., "Fax Number"), dropna() will delete the entire customer record. You might lose valuable data in the "Revenue" column just because the "Fax Number" was missing.
To refine this, you can use the subset parameter:
```python
# Only drop rows where 'Revenue' is missing. 
# Keep the row even if 'Sector' is missing.
valid_revenue_data = crm_data.dropna(subset=['Revenue'])
```

Dropping Columns Sometimes, the problem isn't the observation (row); it's the feature (column). If you ingest a dataset where the "Second Phone Number" column is 95% blank, imputing it is impossible, and deleting the rows would leave you with no data. The solution is to drop the column entirely.
```python
# Drop columns (axis=1) that have any missing values
# Note: In practice, you usually drop specific columns by name using .drop()
crm_data_trimmed = crm_data.dropna(axis=1)
```

The Trade-off: Deletion ensures that the data you do analyze is real and observed. However, it reduces your sample size (statistical power) and can introduce bias if the data is not "missing at random" (e.g., if only unhappy customers skip the "Satisfaction Score" question, deleting those rows makes your customers look happier than they are).
Strategy 2: Imputation (Filling in the Blanks)
Imputation involves replacing missing data with a substitute value based on other available information. In Pandas, we use the .fillna() method.
Constant Imputation This is common for categorical data or specific business logic. If a customer has no "Assigned Sales Rep," you might fill that blank with "Unassigned." If a "Discount Code" field is empty, you might assume the discount is 0.
```python
# Fill missing Sector values with 'Unknown'
crm_data['Sector'] = crm_data['Sector'].fillna('Unknown')


# Fill missing Revenue with 0 (Use with caution!)
crm_data['Revenue'] = crm_data['Revenue'].fillna(0)
 A flowchart decision tree. The top box asks "Is the missing value numerical or categorical?". The Categorical path leads to "Fill with 'Unknown' or Mode". The Numerical path splits into "Time Series" (Forward Fill) and "General" (Mean/Median Imputation). 

```

A flowchart decision tree. The top box asks "Is the missing value numerical or categorical?". The Categorical path leads to "Fill with 'Unknown' or Mode". The Numerical path splits into "Time Series" (Forward Fill) and "General" (Mean/Median Imputation).
Statistical Imputation (Mean vs. Median) For numerical data, filling with 0 is often dangerous. If you are analyzing the average height of adults, filling missing values with 0 will drastically drag down your average.
Instead, we usually fill the gap with the Mean (average) or Median (middle value) of that column.
* Mean Imputation: Best for normally distributed data (bell curve).
* Median Imputation: Best for data with outliers (skewed data).
Consider our Employee_Count. If most companies have 50 employees, but one has 50,000, the average will be artificially high. The median is safer for things like salaries, house prices, or company sizes.
```python
# Calculate the median of existing values
median_employees = crm_data['Employee_Count'].median()


# Fill the missing values with that calculated median
crm_data['Employee_Count'] = crm_data['Employee_Count'].fillna(median_employees)
```

Context-Aware Imputation (Grouping) This is the "Pro" move. In Excel, you might fill a missing value with the overall average. In Python, you can be more specific.
Imagine a missing "Salary" field. Instead of filling it with the average salary of everyone, you can fill it with the average salary of people with the same Job Title.
```python
# Concept example: Fill missing salary with the average salary of that specific job title
df['Salary'] = df.groupby('Job_Title')['Salary'].transform(
    lambda x: x.fillna(x.mean())
)
```

This creates a much more accurate guess than a blanket average.
Special Case: Time Series Data If you are analyzing stock prices or daily temperature, the "average" is not a good guess for a missing day. If you are missing data for Tuesday, the best guess is usually whatever happened on Monday.
This is called Forward Fill (ffill).
```python
# If data is missing, take the value from the previous row
stock_data['Price'] = stock_data['Price'].fillna(method='ffill')
```

Summary: Which Strategy to Use?
| Scenario | Strategy | Pandas Method | | :--- | :--- | :--- | | Rows with crucial missing data | Deletion | df.dropna(subset=['Column']) | | Columns with > 50% missing data | Deletion | df.drop(columns=['Column']) | | Categorical data (e.g., Region) | Constant Imputation | df.fillna('Unknown') | | Numerical data (Normal distribution) | Mean Imputation | df.fillna(df.mean()) | | Numerical data (With outliers) | Median Imputation | df.fillna(df.median()) | | Time Series / Sequential data | Forward Fill | df.fillna(method='ffill') |
Handling missing data is more art than science. It requires understanding why the data is missing. Always document your decision. If you choose to fill missing revenue with the median, state that clearly in your final report, as it fundamentally changes the nature of the dataset.
Data Type Conversion and Formatting Consistency
In Excel, data types are often a suggestion rather than a rule. You can type a date into cell A1, a currency figure into A2, and a text comment into A3, and Excel won’t complain. It applies formatting dynamically, often guessing what you intend. If you type "100" and "200" as text, Excel might still helpfully sum them up to "300" if you use a formula.
In Python, however, types are strict.
A common frustration for professionals moving to Data Science occurs during their first aggregation attempt. You load a sales dataset, group by region, and try to sum the Revenue column. Instead of a dollar amount, you get an error, or worse, a concatenation of strings like 10002000500.
This happens because Pandas is treating your numbers as text. This section covers how to audit, force, and fix data types to ensure your analysis rests on a solid mathematical foundation.
The "Object" Trap
When Pandas loads a CSV file, it scans the data to determine what type belongs in each column. If a column contains integers, it assigns an int64 type. If it sees decimals, it assigns float64.
However, if a column contains a mix of numbers and strings—or if your numbers contain non-numeric characters like currency symbols ($), commas (,), or percentage signs (%)—Pandas defaults to the safest, most flexible type available: object.
In the Pandas world, object is synonymous with "string" or "mixed data."
 A visual comparison between two columns. Left Column: "Excel Style" showing mixed content (numbers with dollar signs, text). Right Column: "Pandas Style" showing the underlying data type storage. The Excel column looks nice but is messy; the Pandas column is labeled 'Dtype: Object' and highlights that mathematical operations are blocked. 

A visual comparison between two columns. Left Column: "Excel Style" showing mixed content (numbers with dollar signs, text). Right Column: "Pandas Style" showing the underlying data type storage. The Excel column looks nice but is messy; the Pandas column is labeled 'Dtype: Object' and highlights that mathematical operations are blocked.
Before doing any analysis, you must check your types using the .info() or .dtypes attribute.
```python
import pandas as pd


# A sample dataset representing common "dirty" business data
data = {
    'Customer_ID': [101, 102, 103, 104],
    'Join_Date': ['2023-01-15', '2023/02/10', 'Mar 1, 2023', '2023-04-20'],
    'Sales_Amount': ['$1,000.00', '$2,500.50', 'Pending', '$450.00'],
    'Department': ['Sales ', 'sales', 'Marketing', ' Sales']
}


df = pd.DataFrame(data)


print(df.dtypes)
```

**Output:**

```

```text
Customer_ID      int64
Join_Date       object
Sales_Amount    object
Department      object
dtype: object
```

Notice that Sales_Amount is an object. You cannot sum this column yet.
Converting Numbers: Stripping and Forcing
To convert Sales_Amount to a number, you cannot simply command Python to "make it a number." Python doesn't know what to do with the $ or the ,. You must first clean the strings, then convert the type.
We use the .astype() method for clean conversions and pd.to_numeric() for messier situations.
Step 1: String Manipulation We access string methods in Pandas using the .str accessor. Here, we replace symbols with nothing (empty strings).
```python
# Remove '$' and ',' from the column
df['Sales_Amount_Clean'] = df['Sales_Amount'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
```

Step 2: Handling Non-Numeric Values Even after removing symbols, our dataset contains the word "Pending". If we try to convert "Pending" to a number, Python will crash.
We use pd.to_numeric with the errors='coerce' argument. This argument tells Pandas: "If you find something that isn't a number, don't crash; just turn it into NaN (Not a Number)."
 A flowchart illustrating the 'pd.to_numeric' logic. Input: A list ["100", "200", "Pending"]. Processing: 'errors=coerce' acts as a filter. Output: The list becomes [100.0, 200.0, NaN], with "Pending" dropping into a waste bin labeled 'Missing Data'. 

A flowchart illustrating the 'pd.to_numeric' logic. Input: A list ["100", "200", "Pending"]. Processing: 'errors=coerce' acts as a filter. Output: The list becomes [100.0, 200.0, NaN], with "Pending" dropping into a waste bin labeled 'Missing Data'.
```python
# Convert to numeric, turning "Pending" into NaN
df['Sales_Amount_Clean'] = pd.to_numeric(df['Sales_Amount_Clean'], errors='coerce')


print(df[['Sales_Amount', 'Sales_Amount_Clean']])
print(df.dtypes)
```

Now, Sales_Amount_Clean is a float64. You can calculate the mean, sum, or standard deviation.
The Datetime Standard
In Excel, dates are actually serial numbers formatted to look like dates. In Pandas, we convert date-like strings into Timestamp objects. This allows us to easily extract the year, month, or day, and perform time-series logic (e.g., "subtract 30 days from today").
In our sample data, Join_Date has mixed formats (2023-01-15 vs Mar 1, 2023). Pandas is surprisingly intelligent at parsing these using pd.to_datetime.
```python
# Convert the Join_Date column to datetime objects
df['Join_Date'] = pd.to_datetime(df['Join_Date'])


# Now we can extract features easily
df['Month'] = df['Join_Date'].dt.month_name()


print(df[['Join_Date', 'Month']])
```

Note: If your dates are in a very specific non-standard format (like `15012023` for Jan 15, 2023), you may need to provide a format string, similar to Excel custom formatting, using the `format` argument.
String Consistency: The Silent Grouping Killer
One of the most insidious issues in data cleaning is string inconsistency. To a human, "Sales", "sales", and " Sales " (with a leading space) are the same department. To a computer, they are three distinct categories.
If you run a groupby on the Department column in our sample data without cleaning it, you will get three separate rows for Sales.
 A diagram showing a GroupBy operation failing due to string inconsistencies. Three buckets labeled "Sales", "sales", and " Sales " collect data separately. An arrow points to a "Unified Bucket" labeled "sales" showing how cleaning merges them. 

A diagram showing a GroupBy operation failing due to string inconsistencies. Three buckets labeled "Sales", "sales", and " Sales " collect data separately. An arrow points to a "Unified Bucket" labeled "sales" showing how cleaning merges them.
To fix this, we standardize casing and remove whitespace.
```python
# 1. Make everything lowercase
# 2. Strip whitespace from the start and end
df['Department'] = df['Department'].str.lower().str.strip()


# Now verify the unique values
print(df['Department'].value_counts())
```

**Output:**

```

```text
sales        3
marketing    1
```

Name: Department, dtype: int64
By chaining .str.lower() and .str.strip(), we have collapsed three fragmented categories into a single, authoritative "sales" group.
Summary of Type Conversion
As you prepare your data for analysis, your checklist for consistency should look like this:
1. Check `dtypes` immediately: Don't assume numbers are numbers. 2. Clean then Convert: Remove currency symbols and commas before converting to numeric. 3. Coerce Errors: Use pd.to_numeric(..., errors='coerce') to handle messy data points without stopping your script. 4. Standardize Strings: Always strip whitespace and unify capitalization on categorical text columns before aggregating.
With your data types strictly defined and formatted, your DataFrames are no longer just fragile tables—they are structured datasets ready for heavy analysis.
String Manipulation: Cleaning Textual Categories and Names
In the previous section, we tackled the strict nature of Python data types. We ensured that our sales figures are recognized as floats and our transaction dates are actually timestamps, not strings of text.
However, simply having a column typed as object (text) does not mean the data inside it is ready for analysis. In fact, text data is notoriously messy. Human entry introduces typos, capitalization inconsistencies, and invisible whitespace.
Consider a scenario where you want to group sales by the "Department" column. In Excel, you might see a Dropdown list containing: `Marketing` marketing `Marketing ` (note the trailing space) Mktg
To a human, these refer to the same department. To a computer (and to Python), these are four completely unique values. If you run a groupby aggregation on this column, you won't get one total for Marketing; you will get four fragmented totals.
In Excel, you would fix this using a combination of TRIM(), PROPER(), and "Find & Replace." In Pandas, we handle this through the vectorized string accessor: .str.
The .str Accessor: Your Text Toolkit
When working with a single string in standard Python, you can use methods like "text".upper(). However, a Pandas Series (a column) is not a string; it is a container of strings.
If you try to run df['Department'].upper(), Python will throw an error because the list itself doesn't have an upper method. You need to tell Pandas to look inside the container and apply the logic to every row. We do this by accessing the .str library attached to the series.
 A diagram showing a Pandas Series column on the left with mixed casing and whitespace. An arrow labeled ".str accessor" points to the right, showing the methods .upper(), .strip(), and .split() being applied to each individual cell simultaneously. 

A diagram showing a Pandas Series column on the left with mixed casing and whitespace. An arrow labeled ".str accessor" points to the right, showing the methods .upper(), .strip(), and .split() being applied to each individual cell simultaneously.
Standardization: Case and Whitespace
The two most common reasons for "duplicate" categories are inconsistent capitalization and invisible whitespace (spaces at the beginning or end of a cell).
Let's look at a sample dataset of client names:
```python
import pandas as pd


data = {
    'Client_Name': ['Acme Corp', 'acme corp', 'Acme Corp ', 'Globex', 'GLOBEX '],
    'Contract_ID': [101, 102, 103, 104, 105]
}
df = pd.read_csv('clients.csv') # Assuming we loaded this data
```

If we count the unique values here, Python sees five distinct clients. To fix this, we standardize the text.
Changing Case Just like Excel's =UPPER(), =LOWER(), and =PROPER() functions, Pandas offers: `.str.lower()`: Converts to lowercase. .str.upper(): Converts to uppercase. * .str.title(): Capitalizes the first letter of each word.
Removing Whitespace The "invisible enemy" in data science is the trailing space. It often occurs when data is exported from legacy SQL databases that pad text fields to a fixed length. In Excel, you use =TRIM(). In Pandas, you use .str.strip().
Here is how we clean the client names in one sweep:
```python
# Step 1: Remove whitespace from both ends
df['Client_Name'] = df['Client_Name'].str.strip()


# Step 2: Convert to title case (e.g., "Acme Corp")
df['Client_Name'] = df['Client_Name'].str.title()


# Result: Only two unique clients remain ('Acme Corp' and 'Globex')
```

Replacing Substrings and Cleaning Logic
Sometimes standardization isn't enough. You may need to fix typos, remove specific characters (like currency symbols), or map abbreviations to full names.
In Excel, you might use SUBSTITUTE() or the "Find and Replace" dialog box. In Pandas, we use .str.replace().
Imagine a "Revenue" column that was imported as text because it included the "$" sign and commas (e.g., "$1,200.50"). To convert this to a number, we must first remove the non-numeric characters.
```python
# Remove '$' and ',' from the string
# Note: regex=False tells Python to treat the search strictly as text, not a pattern
df['Revenue_Clean'] = df['Revenue'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)


# Now we can convert the type
df['Revenue_Clean'] = df['Revenue_Clean'].astype(float)
```

You can also use replace to fix categorical inconsistencies, such as changing "Mktg" to "Marketing":
```python
df['Department'] = df['Department'].str.replace('Mktg', 'Marketing')
 A "Before and After" table visualization. The left side shows a dirty dataset with symbols, abbreviations, and inconsistent casing. The right side shows the clean dataset after .str.replace operations, highlighting the specific changes in a contrasting color. 

```

A "Before and After" table visualization. The left side shows a dirty dataset with symbols, abbreviations, and inconsistent casing. The right side shows the clean dataset after .str.replace operations, highlighting the specific changes in a contrasting color.
Splitting Text: The "Text-to-Columns" Equivalent
One of the most beloved features in Excel is the Text-to-Columns wizard, which allows you to split a "Full Name" column into "First Name" and "Last Name" based on a delimiter (like a comma or space).
In Pandas, we achieve this with .str.split().
Consider a column Location formatted as "City, State" (e.g., "Austin, TX").
```python
# This creates a list inside the cell: ['Austin', 'TX']
df['Location'].str.split(',')
```

However, we usually want these in separate columns, not a list. We use the expand=True argument to separate the results into a new DataFrame structure.
```python
# Split into two new columns
df[['City', 'State']] = df['Location'].str.split(',', expand=True)


# Clean up the whitespace that might be left after the comma in ' State'
df['State'] = df['State'].str.strip()
String Slicing
```

Sometimes you don't need to split by a delimiter; you need to extract a specific number of characters. In Excel, you use =LEFT(cell, 2) or =RIGHT(cell, 4).
In Pandas, because the .str accessor treats the column like a Python string, you can use slicing directly.
* Excel: =LEFT(A1, 3)
* Pandas: df['col'].str[:3]
* Excel: =RIGHT(A1, 2)
* Pandas: df['col'].str[-2:]
Summary Checklist
When preparing textual data for aggregation or analysis, run through this mental checklist:
1. Check for Case: Do "Apple" and "apple" exist? Use .str.title() or .str.lower(). 2. Check for Whitespace: Are there phantom spaces? Always run .str.strip(). 3. Check for Artifacts: Are there symbols ($%#) preventing math? Use .str.replace(). 4. Check Structure: Is useful data combined? Use .str.split(expand=True).
By mastering the .str accessor, you transform raw, messy text into structured categories, allowing the GroupBy and Pivot Table operations we discussed earlier to function accurately.
Detecting and Managing Outliers in Financial and Operational Data
In the previous section, we discussed how to standardize text data, ensuring that "New York," "new york," and "NY" are treated as the same entity. We cleaned up the labels of our data. Now, we must look at the values themselves—specifically, the values that look suspicious.
In business analytics, there is a famous joke: "Bill Gates walks into a bar. Suddenly, on average, everyone in the bar is a billionaire."
This highlights the danger of outliers. In financial and operational data, an outlier is a data point that differs significantly from other observations. Sometimes, these are errors (e.g., a cashier entered $1,000,000 instead of $100). Other times, they are legitimate but extreme events (e.g., a "whale" customer placing a massive order).
In Excel, you might spot these by sorting a column from Largest to Smallest and seeing if the top number makes sense. In Python, relying on manual sorting is risky because you can’t eyeball a million rows. Instead, we use statistical rules and visualization to detect these anomalies systematically.
The Danger of the Mean
Before we write code, understand why we do this. Outliers wreak havoc on standard aggregation metrics, particularly the Mean (Average).
If you are analyzing "Time to Ship" for a logistics company, and 99 packages ship in 2 days, but one package gets lost and ships in 200 days, your average shipping time might jump to 4 days. If you report "Average shipping is 4 days" to management, you are misleading them—most customers get their packages in 2 days.
 A comparison chart showing two distributions. On the left, a normal distribution where the Mean and Median are the same. On the right, a skewed distribution (like income or transaction value) where a few high outliers pull the Mean far to the right, while the Median remains representative of the majority. 

A comparison chart showing two distributions. On the left, a normal distribution where the Mean and Median are the same. On the right, a skewed distribution (like income or transaction value) where a few high outliers pull the Mean far to the right, while the Median remains representative of the majority.
Visual Detection: The Boxplot
The fastest way to detect outliers in Python is not a table, but a Boxplot.
A boxplot (or box-and-whisker plot) provides a visual summary of the central tendency and variability of your data. It draws a box around the middle 50% of your data and extends "whiskers" to the rest. Any dots floating beyond those whiskers are statistically considered outliers.
Let's look at a dataset of invoice amounts:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Sample financial data
data = {
    'Invoice_ID': range(1, 11),
    'Amount': [100, 110, 105, 98, 102, 95, 108, 101, 100, 5000] # Note the 5000
}
df = pd.DataFrame(data)


# Visualizing with a Boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Amount'])
plt.title('Distribution of Invoice Amounts')
plt.show()
 A boxplot generated from the code above. The box is squashed on the left side around the 100 mark. A single lonely dot sits far to the right at the 5000 mark. Arrows analyze the plot: identifying the "Interquartile Range" (the box) and the "Outlier" (the dot). 

```

A boxplot generated from the code above. The box is squashed on the left side around the 100 mark. A single lonely dot sits far to the right at the 5000 mark. Arrows analyze the plot: identifying the "Interquartile Range" (the box) and the "Outlier" (the dot).
In the resulting plot, the "box" will be squashed near 100, and a single dot will sit far away at 5000. That dot is your anomaly.
Statistical Detection: The IQR Method
Visuals are great for exploration, but you cannot automate a pipeline based on looking at pictures. You need a mathematical rule to define what counts as an "outlier."
In Data Science, the standard industry method for non-normal data (like prices or salaries) is the Interquartile Range (IQR) Method.
Here is the logic, which mimics how the boxplot is constructed: 1. Q1 (25th Percentile): The value below which 25% of the data falls. 2. Q3 (75th Percentile): The value below which 75% of the data falls. 3. IQR: The difference between Q3 and Q1 (the middle 50% of data). 4. The Fence: We calculate "fences" or limits. Any data point outside these fences is an outlier. Lower Limit = $Q1 - (1.5 \times IQR)$ Upper Limit = $Q3 + (1.5 \times IQR)$
Let's apply this logic using Pandas:
```python
# Calculate Q1 and Q3
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)


# Calculate IQR
```

IQR = Q3 - Q1


# Define the bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


print(f"Normal range is between ${lower_bound:.2f} and ${upper_bound:.2f}")


# Filter to find the outliers
outliers = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]
print("Detected Outliers:")
print(outliers)
If you run this code, Python will mathematically confirm that the $5,000 invoice is an outlier because it falls far above the upper_bound.
Managing Outliers: Delete, Keep, or Cap?
Once you have identified the outliers using the code above, you face a business decision. Python cannot make this decision for you; it requires domain knowledge.
1. Removal (Trimming) If the outlier is clearly an error (e.g., a customer age of 150, or a negative price), you should delete it.
```python
# Create a clean DataFrame without outliers
df_clean = df[(df['Amount'] >= lower_bound) & (df['Amount'] <= upper_bound)]
2. Retention If the outlier is real (e.g., a massive B2B deal in a dataset of B2C sales), deleting it means lying about your total revenue. In this case, you might keep it, but analyze it separately. You might create a separate report: "Standard Sales Trends" (excluding outliers) and "Key Account Activity" (outliers only).
3. Capping (Winsorization) This is a common technique in financial modeling. Instead of deleting the data, you "cap" it at a specific threshold. If the upper bound is $\$130$, and you have a value of $\$5,000$, you replace the $\$5,000$ with $\$130$. This preserves the fact that the transaction was "high" without allowing the extreme magnitude to ruin your averages.
```

```python
import numpy as np


# Create a copy to avoid SettingWithCopy warnings
df_capped = df.copy()


# Cap values greater than upper_bound to the upper_bound value
df_capped['Amount'] = np.where(df_capped['Amount'] > upper_bound, 
                               upper_bound, 
                               df_capped['Amount'])


print(df_capped)
```

In this capped dataset, the $\$5,000$ entry becomes roughly $\$126$ (depending on the exact IQR calculation). Your mean is no longer skewed, but you haven't lost the record entirely.
Summary of Logic 1. Visual Check: Use a Boxplot to see if outliers exist. 2. Math Check: Use the IQR method to identify specific rows. 3. Business Decision: Decide if the data is wrong (delete it) or exceptional (cap it or segment it).
[[IMAGE: Decision tree flowchart for handling outliers. Step 1: Is the value impossible? (e.g., Age -5). Yes -> Delete/Impute. No -> Step 2. Step 2: Is it a measurement error? Yes -> Delete/Resample. No -> Step 3. Step 3: Does it skew the model significantly? Yes -> Cap/Transform/Log Scale. No -> Keep as is.]]
By mastering outlier detection, you ensure that the insights you deliver to your stakeholders describe the reality of the business, rather than the distortions caused by a few data anomalies.
