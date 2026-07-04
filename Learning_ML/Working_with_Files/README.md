# 📂 Working with Files

This folder contains my Jupyter notebooks focused on learning how to import, access, collect, and work with data from different file formats, databases, APIs, and websites using Python.

Data ingestion is the first step in every Machine Learning workflow. Before cleaning, analyzing, or modeling data, it is essential to know how to retrieve datasets from multiple sources such as CSV files, JSON documents, relational databases, REST APIs, and websites.

The notebooks in this folder document my learning journey as I explore these different data sources using Pandas and related Python libraries.

---

# 📚 Topics Covered

## 📄 Working with CSV Files

This notebook explores the `pandas.read_csv()` function in depth and demonstrates how various parameters can be used to efficiently import CSV datasets.

### Concepts Covered

* Reading CSV files
* Loading datasets from local storage
* Reading datasets directly from URLs
* Selecting specific columns using `usecols`
* Reading only a subset of rows using `nrows`
* Skipping rows with `skiprows`
* Specifying custom column names
* Setting index columns
* Handling missing values during import
* Parsing date columns
* Changing delimiters using `sep`
* Controlling data types using `dtype`
* Loading large datasets efficiently using `chunksize`
* Basic inspection of imported datasets

### Skills Learned

* Understanding the flexibility of `pd.read_csv()`
* Efficiently importing datasets
* Reducing memory usage while loading data
* Preparing datasets for preprocessing and analysis

---

## 🌐 Working with JSON

This notebook demonstrates how to work with JSON data using Pandas.

### Concepts Covered

* Reading JSON files using `pd.read_json()`
* Importing JSON datasets from online sources
* Understanding JSON structure
* Working with nested JSON data
* Reading JSON responses from REST APIs
* Exploring API-generated datasets

### Example Datasets

* Recipe dataset in JSON format
* Currency Exchange Rate API

### Skills Learned

* Importing JSON datasets
* Accessing API data directly in Python
* Converting JSON responses into Pandas DataFrames
* Preparing semi-structured data for analysis

---

## 🗄️ Working with SQL Databases

This notebook introduces database connectivity using MySQL and Pandas.

### Concepts Covered

* Installing MySQL Connector
* Connecting Python to a MySQL database
* Creating database connections
* Executing SQL queries
* Importing SQL query results into Pandas DataFrames
* Exploring relational database tables

### Libraries Used

* mysql-connector-python
* pandas

### Example SQL Query

```sql
SELECT * FROM countrylanguage;
```

### Skills Learned

* Connecting Python applications to databases
* Retrieving data using SQL
* Integrating SQL query results with Pandas
* Building the foundation for data engineering and Machine Learning workflows

---

## 🕸️ Working with Web Scraping

This notebook introduces the fundamentals of web scraping using Python to collect publicly available data directly from websites.

Instead of reading data from existing files or databases, web scraping enables the extraction of information from HTML pages and converts it into structured datasets suitable for analysis.

### Concepts Covered

* Sending HTTP requests using `requests`
* Using custom request headers
* Parsing HTML using BeautifulSoup
* Exploring webpage structure
* Finding HTML elements
* Extracting text from HTML tags
* Working with nested HTML elements
* Cleaning extracted text
* Creating Pandas DataFrames
* Scraping multiple pages using loops
* Combining scraped data using `pd.concat()`

### Example Website

* AmbitionBox Company Listings

### Skills Learned

* Retrieving webpage content
* Parsing HTML documents
* Extracting structured information
* Navigating webpage elements
* Automating data collection
* Creating datasets from web pages

---

# 🛠️ Technologies Used

* Python
* Jupyter Notebook
* Pandas
* Requests
* BeautifulSoup4
* lxml
* MySQL
* mysql-connector-python

---

# 📦 Libraries Used

* pandas
* requests
* beautifulsoup4
* lxml
* mysql-connector-python

---

# 📌 Key Functions Explored

## CSV

* `pd.read_csv()`

Important parameters explored:

* `filepath_or_buffer`
* `sep`
* `header`
* `names`
* `index_col`
* `usecols`
* `skiprows`
* `nrows`
* `dtype`
* `parse_dates`
* `na_values`
* `chunksize`

---

## JSON

* `pd.read_json()`

---

## SQL

* `mysql.connector.connect()`
* `pd.read_sql_query()`

---

## Web Scraping

### Requests

* `requests.get()`

### BeautifulSoup

* `BeautifulSoup()`
* `find()`
* `find_all()`
* `prettify()`
* `.text`
* `.strip()`

### Pandas

* `pd.DataFrame()`
* `pd.concat()`

---

# 🎯 Learning Outcomes

Through these notebooks, I learned how to:

* Import datasets from multiple file formats
* Load structured and semi-structured data into Pandas
* Read data directly from online sources
* Consume REST API responses
* Connect Python to relational databases
* Execute SQL queries from Python
* Retrieve webpage content using HTTP requests
* Parse HTML documents
* Extract useful information from websites
* Automate data collection across multiple webpages
* Convert scraped data into DataFrames
* Prepare datasets for preprocessing, exploratory data analysis (EDA), and machine learning

---

# 🚀 Future Topics

This folder will continue to grow as I learn to work with additional data sources and data collection techniques, including:

* Excel Files
* XML Files
* YAML Files
* Parquet Files
* Feather Files
* Pickle Files
* SQLite
* PostgreSQL
* MongoDB
* APIs with Authentication
* GraphQL APIs
* Web Scraping with Selenium
* Web Scraping with Playwright
* Dynamic Website Scraping
* XPath
* CSS Selectors
* Cloud Storage (AWS S3, Google Cloud Storage)
* Data Warehouses

---

# ⚠️ Ethical Data Collection

When collecting data from websites, it is important to follow responsible scraping practices.

This includes:

* Respecting website Terms of Service
* Reviewing `robots.txt` where applicable
* Avoiding excessive requests
* Using reasonable request intervals
* Collecting only publicly available information
* Using data responsibly and ethically

---

# 💡 Why This Matters

Real-world Machine Learning projects rarely begin with clean CSV files. Data is commonly stored across databases, APIs, cloud storage, websites, and many other file formats.

Developing the ability to efficiently retrieve, collect, and integrate data from these diverse sources is a fundamental skill for every Data Scientist and Machine Learning Engineer.

These notebooks document my progress toward building that foundation through hands-on practice, experimentation, and continuous learning. As I expand this repository, I aim to explore more advanced data ingestion techniques and strengthen the skills required for real-world machine learning projects.
