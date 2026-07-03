# 📂 Working with Files

This folder contains my Jupyter notebooks focused on learning how to import, access, and work with data stored in different file formats and data sources using Python.

Data ingestion is the first step in every Machine Learning workflow. Before cleaning, analyzing, or modeling data, it is essential to know how to retrieve datasets from multiple sources such as CSV files, JSON documents, APIs, and relational databases.

The notebooks in this folder document my learning journey as I explore these different data sources using Pandas and related Python libraries.

---

# 📚 Topics Covered

## 📄 Working with CSV Files

This notebook explores the `pandas.read_csv()` function in depth and demonstrates how various parameters can be used to efficiently import CSV datasets.

### Concepts Covered

- Reading CSV files
- Loading datasets from local storage
- Reading datasets directly from URLs
- Selecting specific columns using `usecols`
- Reading only a subset of rows using `nrows`
- Skipping rows with `skiprows`
- Specifying custom column names
- Setting index columns
- Handling missing values during import
- Parsing date columns
- Changing delimiters using `sep`
- Controlling data types using `dtype`
- Loading large datasets efficiently using `chunksize`
- Basic inspection of imported datasets

### Skills Learned

- Understanding the flexibility of `pd.read_csv()`
- Efficiently importing datasets
- Reducing memory usage while loading data
- Preparing datasets for preprocessing and analysis

---

## 🌐 Working with JSON

This notebook demonstrates how to work with JSON data using Pandas.

### Concepts Covered

- Reading JSON files using `pd.read_json()`
- Importing JSON datasets from online sources
- Understanding JSON structure
- Working with nested JSON data
- Reading JSON responses from REST APIs
- Exploring API-generated datasets

### Example Datasets

- Recipe dataset in JSON format
- Currency Exchange Rate API

### Skills Learned

- Importing JSON datasets
- Accessing API data directly in Python
- Converting JSON responses into Pandas DataFrames
- Preparing semi-structured data for analysis

---

## 🗄️ Working with SQL Databases

This notebook introduces database connectivity using MySQL and Pandas.

### Concepts Covered

- Installing MySQL Connector
- Connecting Python to a MySQL database
- Creating database connections
- Executing SQL queries
- Importing SQL query results into Pandas DataFrames
- Exploring relational database tables

### Libraries Used

- mysql-connector-python
- pandas

### Example SQL Query

```sql
SELECT * FROM countrylanguage;
```

### Skills Learned

- Connecting Python applications to databases
- Retrieving data using SQL
- Integrating SQL query results with Pandas
- Building the foundation for data engineering and Machine Learning workflows

---

# 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- MySQL
- mysql-connector-python

---

# 📌 Key Functions Explored

## CSV

- `pd.read_csv()`

Important parameters explored:

- `filepath_or_buffer`
- `sep`
- `header`
- `names`
- `index_col`
- `usecols`
- `skiprows`
- `nrows`
- `dtype`
- `parse_dates`
- `na_values`
- `chunksize`

---

## JSON

- `pd.read_json()`

---

## SQL

- `mysql.connector.connect()`
- `pd.read_sql_query()`

---

# 🎯 Learning Outcomes

Through these notebooks, I learned how to:

- Import datasets from multiple file formats
- Load structured and semi-structured data into Pandas
- Read data directly from online sources
- Consume REST API responses
- Connect Python to relational databases
- Execute SQL queries from Python
- Convert database query results into DataFrames
- Prepare datasets for preprocessing, exploratory data analysis (EDA), and machine learning

---

# 🚀 Future Topics

This folder will continue to grow as I learn to work with additional data sources, including:

- Excel Files
- XML Files
- Parquet Files
- Feather Files
- Pickle Files
- SQLite
- PostgreSQL
- MongoDB
- APIs with Authentication
- Web Scraping
- Cloud Storage (AWS S3, Google Cloud Storage)
- Data Warehouses

---

# 💡 Why This Matters

Real-world Machine Learning projects rarely begin with clean CSV files. Data is commonly stored across databases, APIs, cloud storage, and various file formats.

Developing the ability to efficiently retrieve and integrate data from these diverse sources is a fundamental skill for every Data Scientist and Machine Learning Engineer. These notebooks document my progress toward building that foundation through hands-on practice and experimentation.