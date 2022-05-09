## Customer Segmentation
Customer segmentation is a method for grouping customers into similar clusters. The machine learning technique used for customer segmentation in this exercise is KMeans Algorithm.

### Data Description
The dataset has the following labels:

| Variable               | Description                         |
|------------------------|-------------------------------------|
| Customer ID            | UniqueID                            |
| Gender                 | Value can be either male or female  |
| Age                    | Customer's age                      |
| Annual Income (k$)     | Customer's annual income in dollars |
| Spending Score (1-100) | Ranges from 1 to 100                |

## Project Structure

```
.
├── data   
│   └── Mall_Customers.csv
├── kmeans-algorithm.py
├── README.md
├── requirements.txt
```

## Usage

```
python3 kmeans-algorithm.py 
```

## License
This project is licensed under the terms of the [MIT license](https://choosealicense.com/licenses/mit/).
