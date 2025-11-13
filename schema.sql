-- Products Table
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(255),
    unit_price DECIMAL(10, 2),
    category VARCHAR(100)
);

-- Regions Table
CREATE TABLE regions (
    region_id INT PRIMARY KEY,
    region_name VARCHAR(100)
);

-- Customers Table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(255),
    region_id INT,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

-- Sales Table
CREATE TABLE sales (
    sale_id INT PRIMARY KEY,
    customer_id INT,
    region_id INT,
    sale_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);
CREATE INDEX sales__sale_date ON sales(sale_date);

-- Transaction Table
CREATE TABLE transactions (
    transaction_id PRIMARY KEY,
    sale_id INT,
    product_id INT,
    units INT,
    FOREIGN KEY (sale_id) REFERENCES sales(sale_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
