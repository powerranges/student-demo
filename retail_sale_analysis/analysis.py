
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the retail sales dataset
df = pd.read_csv("data/01_retail_sales.csv")

# Display first five records
print("First 5 Rows:")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# STEP 2: DATA CLEANING AND PREPROCESSING
# 1. Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())
# 2. Remove duplicate records
df = df.drop_duplicates()
print("\nDataset Shape After Removing Duplicates:")
print(df.shape)
# 3. Handle missing values in City column
df["City"] = df["City"].fillna("Unknown")
print("\nMissing Values After Handling City:")
print(df.isnull().sum())

# 4. Inspect date format
print("\nSample Order Dates:")
print(df["Order_Date"].head(10).to_list())

# 5. Convert Order_Date into datetime format
# Assumes dates are DD/MM/YYYY or YYYY-MM-DD
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce",
    dayfirst=True
)

# 6. Convert numerical columns into numeric format
numeric_columns = ["Quantity", "Unit_Price", "Revenue"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 7. Display updated data types
print("\nData Types After Cleaning:")
print(df.dtypes)

# 8. Final dataset verification
print("\nFINAL CLEANED DATASET")
print("----------------------")

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nFirst 5 Cleaned Records:")
print(df.head())

# 9. Save cleaned dataset
import os
os.makedirs("output", exist_ok=True)
df.to_csv("output/cleaned_retail_sales.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# 1. Calculate overall sales performance

total_revenue = df["Revenue"].sum()
total_orders = df["Order_ID"].nunique()
total_quantity = df["Quantity"].sum()
total_customers = df["Customer_ID"].nunique()

average_order_value = total_revenue / total_orders if total_orders else 0

print("\n========== OVERALL SALES PERFORMANCE ==========")

print("Total Revenue:", total_revenue)
print("Total Orders:", total_orders)
print("Total Quantity Sold:", total_quantity)
print("Total Customers:", total_customers)
print("Average Order Value:", round(average_order_value, 2))


# 2. Revenue by Product Category

category_sales = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)

print("\n========== REVENUE BY CATEGORY ==========")
print(category_sales)

print("\nHighest Revenue Category:", category_sales.idxmax())


# 3. Revenue by City

city_sales = df.groupby("City")["Revenue"].sum().sort_values(ascending=False)

print("\n========== REVENUE BY CITY ==========")
print(city_sales)

print("\nTop Performing City:", city_sales.idxmax())


# 4. Best Selling Products

product_sales = df.groupby("Product")["Quantity"].sum().sort_values(ascending=False)

print("\n========== BEST SELLING PRODUCTS ==========")
print(product_sales)

print("\nBest Selling Product:", product_sales.idxmax())


# 5. Revenue by Payment Mode

payment_sales = df.groupby("Payment_Mode")["Revenue"].sum().sort_values(ascending=False)

print("\n========== REVENUE BY PAYMENT MODE ==========")
print(payment_sales)


# 6. Monthly Sales Analysis

df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("Month")["Revenue"].sum()

print("\n========== MONTHLY SALES ==========")
print(monthly_sales)


# ==========================================
# STEP 4: DATA VISUALIZATION
# ==========================================

# 7. Revenue by Category - Bar Chart

plt.figure(figsize=(10, 5))

sns.barplot(x=category_sales.index, y=category_sales.values)

plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 8. Revenue by City - Bar Chart

plt.figure(figsize=(10, 5))

sns.barplot(x=city_sales.index, y=city_sales.values)

plt.title("Revenue by City")
plt.xlabel("City")
plt.ylabel("Total Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 9. Best Selling Products - Bar Chart

plt.figure(figsize=(10, 5))

sns.barplot(x=product_sales.index, y=product_sales.values)

plt.title("Quantity Sold by Product")
plt.xlabel("Product")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 10. Monthly Revenue Trend - Line Chart

plt.figure(figsize=(12, 5))

sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o")

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 11. Payment Mode Distribution - Pie Chart

payment_count = df["Payment_Mode"].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(
    payment_count.values,
    labels=payment_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Payment Mode Distribution")
plt.show()


print("\nEDA AND DATA VISUALIZATION COMPLETED SUCCESSFULLY!")


# ==========================================
# STEP 4: ADVANCED SALES ANALYSIS
# ==========================================

print("\n================================")
print("ADVANCED SALES ANALYSIS")
print("================================")

# 1. TOP 10 PRODUCTS BY REVENUE
top_products = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products by Revenue:")
print(top_products)


# 2. TOP 10 CITIES BY REVENUE
top_cities = (
    df.groupby("City")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Cities by Revenue:")
print(top_cities)


# 3. CATEGORY-WISE SALES CONTRIBUTION
category_revenue = df.groupby("Category")["Revenue"].sum()

category_contribution = (
    category_revenue / category_revenue.sum()
) * 100

print("\nCategory-wise Sales Contribution (%):")
print(category_contribution.round(2))


# 4. MONTHLY SALES ANALYSIS
monthly_revenue = (
    df.groupby("Month")["Revenue"]
    .sum()
    .sort_index()
)

print("\nMonthly Revenue:")
print(monthly_revenue)


# 5. HIGHEST AND LOWEST SALES MONTH
if not monthly_revenue.empty:
    highest_month = monthly_revenue.idxmax()
    lowest_month = monthly_revenue.idxmin()

    print("\nHighest Sales Month:", highest_month)
    print("Highest Monthly Revenue:", monthly_revenue.max())

    print("\nLowest Sales Month:", lowest_month)
    print("Lowest Monthly Revenue:", monthly_revenue.min())


# 6. PAYMENT MODE REVENUE ANALYSIS
payment_revenue = (
    df.groupby("Payment_Mode")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by Payment Mode:")
print(payment_revenue)


# 7. QUANTITY SOLD BY CATEGORY
category_quantity = (
    df.groupby("Category")["Quantity"]
    .sum()
    .sort_values(ascending=False)
)

print("\nQuantity Sold by Category:")
print(category_quantity)


# 8. OVERALL SALES METRICS
total_revenue = df["Revenue"].sum()
total_quantity = df["Quantity"].sum()
total_customers = df["Customer_ID"].nunique()
total_orders = df["Order_ID"].nunique()

average_order_value = (
    total_revenue / total_orders if total_orders > 0 else 0
)

print("\nOVERALL SALES METRICS")
print("---------------------")
print("Total Revenue:", round(total_revenue, 2))
print("Total Quantity Sold:", total_quantity)
print("Total Customers:", total_customers)
print("Unique Order IDs:", total_orders)
print("Average Revenue per Order ID:", round(average_order_value, 2))


# 9. SAVE ADVANCED ANALYSIS RESULTS
import os

os.makedirs("output", exist_ok=True)

top_products.to_csv("output/top_products.csv")
top_cities.to_csv("output/top_cities.csv")
category_contribution.to_csv("output/category_contribution.csv")
monthly_revenue.to_csv("output/monthly_revenue.csv")
payment_revenue.to_csv("output/payment_revenue.csv")

print("\nAdvanced analysis results saved in the output folder.")

# ==========================================
# STEP 5: DATA VISUALIZATION
# ==========================================

print("\n================================")
print("DATA VISUALIZATION")
print("================================")

import os

os.makedirs("output/charts", exist_ok=True)

# 1. CATEGORY-WISE REVENUE
plt.figure(figsize=(10, 6))

sns.barplot(
    x=category_revenue.index,
    y=category_revenue.values
)

plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/charts/category_revenue.png")
plt.show()


# 2. TOP 10 PRODUCTS
plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("output/charts/top_products.png")
plt.show()


# 3. CITY-WISE REVENUE
plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_cities.values,
    y=top_cities.index
)

plt.title("Top 10 Cities by Revenue")
plt.xlabel("Revenue")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("output/charts/city_revenue.png")
plt.show()


# 4. MONTHLY REVENUE TREND
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_revenue.index,
    monthly_revenue.values,
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("output/charts/monthly_revenue.png")
plt.show()


# 5. PAYMENT MODE REVENUE
plt.figure(figsize=(8, 8))

plt.pie(
    payment_revenue.values,
    labels=payment_revenue.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Revenue Distribution by Payment Mode")
plt.tight_layout()
plt.savefig("output/charts/payment_mode.png")
plt.show()


# 6. QUANTITY SOLD BY CATEGORY
plt.figure(figsize=(10, 6))

sns.barplot(
    x=category_quantity.index,
    y=category_quantity.values
)

plt.title("Quantity Sold by Category")
plt.xlabel("Category")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/charts/category_quantity.png")
plt.show()


# 7. INTERACTIVE MONTHLY REVENUE CHART
fig = px.line(
    monthly_revenue.reset_index(),
    x="Month",
    y="Revenue",
    title="Interactive Monthly Revenue Trend",
    markers=True
)

fig.write_html("output/charts/interactive_monthly_revenue.html")
fig.show()


# 8. INTERACTIVE CATEGORY CHART
category_chart_data = category_revenue.reset_index()

fig = px.bar(
    category_chart_data,
    x="Category",
    y="Revenue",
    title="Interactive Category-wise Revenue",
    color="Category"
)

fig.write_html("output/charts/interactive_category_revenue.html")
fig.show()

print("\nAll charts saved successfully!")

# ==========================================
# STEP 7: FINAL PROJECT REPORT
# ==========================================

print("\n================================")
print("FINAL PROJECT REPORT")
print("================================")

summary = {
    "Metric": [
        "Total Revenue",
        "Total Quantity Sold",
        "Total Customers",
        "Unique Orders",
        "Average Revenue per Order ID",
        "Best Selling Product",
        "Best Performing City",
        "Best Performing Category",
        "Best Sales Month"
    ],

    "Value": [
        round(total_revenue, 2),
        total_quantity,
        total_customers,
        total_orders,
        round(average_order_value, 2),
        top_products.idxmax() if not top_products.empty else "N/A",
        top_cities.idxmax() if not top_cities.empty else "N/A",
        category_revenue.idxmax() if not category_revenue.empty else "N/A",
        monthly_revenue.idxmax() if not monthly_revenue.empty else "N/A"
    ]
}

summary_df = pd.DataFrame(summary)

print(summary_df.to_string(index=False))

summary_df.to_csv(
    "output/final_sales_report.csv",
    index=False
)

print("\nFinal report saved successfully!")
print("\nPROJECT ANALYSIS COMPLETED!")

# ==========================================
# STEP 7: FINAL PROJECT REPORT
# ==========================================

print("\n================================")
print("FINAL PROJECT REPORT")
print("================================")

summary = {
    "Metric": [
        "Total Revenue",
        "Total Quantity Sold",
        "Total Customers",
        "Unique Orders",
        "Average Revenue per Order ID",
        "Best Selling Product",
        "Best Performing City",
        "Best Performing Category",
        "Best Sales Month"
    ],

    "Value": [
        round(total_revenue, 2),
        total_quantity,
        total_customers,
        total_orders,
        round(average_order_value, 2),
        top_products.idxmax() if not top_products.empty else "N/A",
        top_cities.idxmax() if not top_cities.empty else "N/A",
        category_revenue.idxmax() if not category_revenue.empty else "N/A",
        monthly_revenue.idxmax() if not monthly_revenue.empty else "N/A"
    ]
}

summary_df = pd.DataFrame(summary)

print(summary_df.to_string(index=False))

summary_df.to_csv(
    "output/final_sales_report.csv",
    index=False
)

print("\nFinal report saved successfully!")
print("\nPROJECT ANALYSIS COMPLETED!")

# ==========================================
# STEP 7: FINAL PROJECT REPORT
# ==========================================

print("\n================================")
print("FINAL PROJECT REPORT")
print("================================")

summary = {
    "Metric": [
        "Total Revenue",
        "Total Quantity Sold",
        "Total Customers",
        "Unique Orders",
        "Average Revenue per Order ID",
        "Best Selling Product",
        "Best Performing City",
        "Best Performing Category",
        "Best Sales Month"
    ],

    "Value": [
        round(total_revenue, 2),
        total_quantity,
        total_customers,
        total_orders,
        round(average_order_value, 2),
        top_products.idxmax() if not top_products.empty else "N/A",
        top_cities.idxmax() if not top_cities.empty else "N/A",
        category_revenue.idxmax() if not category_revenue.empty else "N/A",
        monthly_revenue.idxmax() if not monthly_revenue.empty else "N/A"
    ]
}

summary_df = pd.DataFrame(summary)

print(summary_df.to_string(index=False))

summary_df.to_csv(
    "output/final_sales_report.csv",
    index=False
)

# ==========================================
# ADDITION 1: REVENUE VERIFICATION
# ==========================================

df["Revenue_Calculated"] = df["Quantity"] * df["Unit_Price"]

df["Revenue_Difference"] = (
    df["Revenue_Calculated"] - df["Revenue"]
)

df["Revenue_Match"] = np.isclose(
    df["Revenue_Calculated"],
    df["Revenue"],
    equal_nan=False
)

print("\n========== REVENUE VERIFICATION ==========")
print("Total Calculated Revenue:", df["Revenue_Calculated"].sum())
print("Total Existing Revenue:", df["Revenue"].sum())
print("Matching Records:", df["Revenue_Match"].sum())
print("Mismatched Records:", (~df["Revenue_Match"]).sum())

# Use calculated revenue for subsequent analysis
# Keep the original Revenue column unchanged
df["Revenue_Analysis"] = df["Revenue_Calculated"]
# ==========================================
# ADDITION 2: DATE ANALYSIS
# ==========================================

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce",
    dayfirst=True
)

df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

df["Weekday"] = df["Order_Date"].dt.day_name()

weekday_sales = df.groupby("Weekday")["Revenue_Analysis"].sum()

print("\n========== WEEKDAY SALES ==========")
print(weekday_sales.sort_values(ascending=False))
# ==========================================
# ADDITION 3: TOP 10 CUSTOMERS
# ==========================================

top_customers = (
    df.groupby("Customer_ID")["Revenue_Analysis"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 CUSTOMERS ==========")
print(top_customers)

top_customers.to_csv("output/top_10_customers.csv")
# ==========================================
# ADDITION 4: PAYMENT METHOD ANALYSIS
# ==========================================

payment_summary = df.groupby("Payment_Mode").agg(
    Total_Orders=("Order_ID", "nunique"),
    Total_Revenue=("Revenue_Analysis", "sum"),
    Total_Quantity=("Quantity", "sum")
).sort_values(
    by="Total_Revenue",
    ascending=False
)

print("\n========== PAYMENT METHOD COMPARISON ==========")
print(payment_summary)

payment_summary.to_csv("output/payment_method_summary.csv")
# ==========================================
# ADDITION 5: IQR OUTLIER DETECTION
# ==========================================

order_revenue = (
    df.groupby("Order_ID")["Revenue_Analysis"]
    .sum()
)

Q1 = np.percentile(order_revenue, 25)
Q3 = np.percentile(order_revenue, 75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

large_orders = order_revenue[
    order_revenue > upper_limit
]

print("\n========== IQR OUTLIER ANALYSIS ==========")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

print("\nUnusually Large Orders:")
print(large_orders)

print("\nNumber of Large Orders:", len(large_orders))

large_orders.to_csv("output/unusually_large_orders.csv")
print("\nFinal report saved successfully!")
print("\nPROJECT ANALYSIS COMPLETED!")