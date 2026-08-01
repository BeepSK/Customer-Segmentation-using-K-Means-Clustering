import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide")


original_df = pd.read_csv("../datasets/store_customers.csv")
df = pd.read_csv("../datasets/Mall_Customers_Clean.csv")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Home",
        "Dataset",
        "EDA",
        "Customer Segmentation",
        "Business Insights"])

if page == "Home":

    st.title("📊 Customer Segmentation Dashboard")

    st.markdown("""
    ## Welcome!

    This dashboard demonstrates **Customer Segmentation using K-Means Clustering**.

    The goal of this project is to analyze customer purchasing behaviour and divide customers into meaningful groups that businesses can target more effectively.
    """)
    
    st.info("""👈 Use the navigation menu in the sidebar to explore the dataset, visualizations, clustering results, and business insights.""")

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze customer behaviour 
    - Preprocess and clean the dataset
    - Perform Exploratory Data Analysis (EDA) 
    - Apply K-Means Clustering
    - Develop an interactive Streamlit dashboard 
    - Generate business insights for decision-making 
    """)
    
    st.header("📈 Project Highlights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Customers", "1000")
    with col2:
        st.metric("Features", "5")
    with col3:
        st.metric("Clusters", "4")
    
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Algorithm", "K-Means")
    with col5:
        st.metric("Dashboard", "Streamlit")
    
    st.header("🛠 Technologies Used")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    """)
    
    with col2:
        st.write("""
    - Matplotlib
    - Seaborn
    - Plotly
    - Streamlit
    """)
    
    st.header("📂 Project Workflow")
    st.info("""
    Dataset → Data Preprocessing → Exploratory Data Analysis → Feature Scaling → K-Means Clustering → Customer Segmentation → Business Insights
    """)
    
    st.divider()
    st.caption("Developed as part of an End-to-End Machine Learning Project using Python and Streamlit.")
        
elif page == "Dataset":

    st.title("📂 Dataset Overview")

    st.write("This page provides an overview of the customer dataset.")

    st.subheader("Source of the Data")
    st.write("For this project, I have used a dataset available on Kaggle titled store customers.csv. This is a dataset commonly used for Customer Segmentation and clustering which makes it a good fit for the project. It includes all important demographic and spending-related information of the customers at the mall.")
    st.write("First and foremost, I had to import the data from the dataset into my Python notebook to start working on it. To do so, I used the pandas library. The dataset was stored in CSV (Comma-Separated Values) format and loaded into a DataFrame for further preprocessing and analysis.")
    
    st.subheader("Dataset Preview")
    st.write("Before doing the analysis, the structure and other important qualities of the dataset were also examined properly. This included identifying missing values and verifying the data types of each feature. Data preprocessing was performed to ensure consistency and improve the quality of the dataset before applying machine learning techniques. The final dataset contains five attributes, including one unique identifier, one categorical feature, and three numerical features. These variables provide the necessary information for performing customer segmentation.")
    st.header("📊 Original Dataset")
    st.write("Before preprocessing, the dataset was inspected to identify missing values, duplicate records, and inconsistencies in the data.")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", original_df.shape[0])

    with col2:
        st.metric("Columns", original_df.shape[1])

    with col3:
        st.metric("Missing Values", original_df.isnull().sum().sum())
        
    with col4:
        st.metric("Duplicate Rows", original_df.duplicated().sum())
    
    st.header("🧹 Cleaned Dataset")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
        
    st.subheader("✅ Data Cleaning Summary")
    st.success("""
    • Missing values handled

    • Duplicate records removed

    • Dataset prepared for machine learning

    • Final dataset ready for clustering
    """)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    info = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values})
    
    st.subheader("📑 Feature Information")
    st.dataframe(info, use_container_width=True)
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    st.write("We have both demographic and behavioural information. Variables such as age, annual income, and spending score allow the clustering algorithm to identify customers with similar purchasing patterns. The insights obtained from these customer groups can assist businesses in designing targeted marketing strategies, improving customer retention, and enhancing overall decision-making.")
    st.write("Furthermore, the dataset provides an appropriate balance between simplicity and practical relevance, making it suitable for demonstrating the application of K-Means clustering in a real-world business context.")

elif page == "EDA":

    st.title("Exploratory Data Analysis")
    
    st.write("Exploratory Data Analysis (EDA) is an essential step in the data science workflow that involves examining and visualizing the dataset to understand its underlying structure, identify patterns, detect anomalies, and summarize the main characteristics of the data.") 
    st.write("In this project, EDA was performed using statistical summaries and graphical visualizations to gain insights into customer demographics and purchasing behavior before applying the K-Means clustering algorithm. The analysis helped identify trends, potential outliers, and relationships among the variables, providing a strong foundation for the clustering process")
    
    st.subheader("Select a Feature")
    feature = st.selectbox(
    "Choose a numerical feature:",
    ["Age", "Annual_Income", "Spending_Score"])
    
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(
    data=df,
    x=feature,
    bins=20,
    kde=True,
    ax=ax)
    ax.set_title(f" {feature} Distribution")
    st.pyplot(fig)
    st.markdown(f"""### 📌 Interpretation
                This histogram shows the distribution of {feature} among all customers.""")
    
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(
    data=df,
    x="Gender",
    ax=ax)
    st.pyplot(fig)
    st.markdown(f""" ### 📌 Interpretation
                Looking at the Gender Distribution Chart helps us understand the number of male and female customers. As can be seen, there is but little difference in their numbers, with female customers leading by a few digits. Information about customers from both genders allows the clustering algorithm to analyze purchasing behavior across a diverse customer base.""")
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax)
    st.pyplot(fig)
    st.markdown(""" ### 📌 Interpretation
                The heatmap illustrates the correlation between numerical features in the dataset. Correlation values close to 1 indicate a strong positive relationship, values close to -1 indicate a strong negative relationship, and values near 0 indicate little or no linear relationship. For this dataset, Age and Annual Income exhibit a strong negative correlation with Spending Score, suggesting that these variables play an important role in distinguishing customer groups.""")
    
    st.subheader("**Key Observation**")
    st.info("""
            • Age has a strong negative correlation with Spending Score.

            • Annual Income also shows a strong negative correlation with Spending Score.

            • These relationships suggest that customer spending behaviour varies significantly across different age and income groups, making these features suitable for clustering.
            """)
            
    st.subheader("Some more Analysis")
    st.write("Using the spending score chart, I noticed that people are either spending a lot of money or they aren’t. I wanted to investigate this relationship further by comparing the spending score with age and income to see if anything comes up.")
    
    st.markdown("#### 👥 Age vs Spending Score")
    st.subheader("Age vs Spending Score")
    fig, ax = plt.subplots(figsize=(8,5))

    sns.scatterplot(
        data=df,
        x="Age",
        y="Spending_Score",
        ax=ax
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Spending Score")

    st.pyplot(fig)

    st.markdown("""
    ### 📌 Interpretation

    This scatter plot illustrates the relationship between customer age and spending score.
    A clear negative trend is observed, indicating that younger customers generally exhibit
    higher spending scores, while spending behaviour tends to decrease with increasing age.
    """)
    corr = df["Age"].corr(df["Spending_Score"])

    st.metric(
        "Correlation (Age vs Spending Score)",
        f"{corr:.3f}")
    st.write("The value -0.835 is very close to -1, which indicates a strong negative linear relationship. This confirms our understanding that as Age increases, Spending Score tends to decrease.")

    st.markdown("#### 💰 Annual Income vs Spending Score")
    st.subheader("Annual Income vs Spending Score")
    fig, ax = plt.subplots(figsize=(8,5))

    sns.scatterplot(
        data=df,
        x="Annual_Income",
        y="Spending_Score",
        ax=ax
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")

    st.pyplot(fig)

    st.markdown("""
    ### 📌 Interpretation

    This scatter plot shows the relationship between annual income and spending score.
    Although customer spending varies across different income levels, a noticeable negative
    trend is present, suggesting that higher annual income does not necessarily correspond
    to higher spending behaviour in this dataset.
    """)
    
    corr = df["Annual_Income"].corr(df["Spending_Score"])

    st.metric(
        "Correlation (Income vs Spending Score)",
        f"{corr:.3f}")
    st.write("As in the case for Age, this means that as Annual Income increases, Spending Score generally decreases.")
    
    st.header("📌 Overall Findings")
    st.info("""
    • Younger customers generally have higher spending scores.

    • Spending behaviour decreases with increasing age.

    • Annual Income alone is not a reliable predictor of customer spending.

    • Male and female customers exhibit similar spending behaviour.

    • These findings justify the use of customer segmentation to identify meaningful customer groups based on multiple features rather than a single variable.
    """)
    
elif page == "Customer Segmentation":

    st.title("🤖 Customer Segmentation")
    st.subheader("Introduction to K-Means Clustering")
    st.write("K-Means is an unsupervised machine learning algorithm widely used for clustering tasks. Unlike supervised learning algorithms, K-Means does not require labeled data. Instead, it partitions observations into a predefined number of clusters based on the similarity of their features. The algorithm aims to minimize the distance between data points and the centroid of the cluster to which they belong. In this project, K-Means clustering was employed to identify groups of customers with similar purchasing behaviour based on their annual income and spending score.")
    
    st.subheader("Feature Selection and Scaling")
    st.write("Selecting appropriate features is a critical step in the clustering process. For this project, Annual Income and Spending Score were chosen as the clustering variables because they directly represent customers’ purchasing capacity and spending behaviour.")
    feature_table = pd.DataFrame({
    "Feature": [
        "Annual Income (k$)",
        "Spending Score (1-100)"],
    "Reason for Selection": [
        "Measures customers' purchasing capacity and exhibited a strong relationship with Spending Score, making it a valuable clustering feature.",
        "Represents customer purchasing behaviour and serves as the primary indicator for distinguishing different customer segments."]})

    st.table(feature_table)
    st.write("Since K-Means is a distance-based algorithm, feature scaling was performed before clustering. Standardization was applied using the StandardScaler technique, which transforms each feature to have a mean of zero and a standard deviation of one. This prevents variables with larger numerical values from dominating the clustering process and ensures that both features contribute equally to the calculation of distances.")
    
    st.subheader("Determining the number of Clusters")
    X = df[["Annual_Income", "Spending_Score"]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(
            n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10)

    df["Cluster"] = kmeans.fit_predict(X_scaled)
        
    st.subheader("Elbow Method")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1,11), wcss, marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS")
    ax.set_title("#### Elbow Method")
    st.pyplot(fig)
    st.write("The Elbow Method was used to determine the optimal value of k by plotting the Within-Cluster Sum of Squares (WCSS) against different numbers of clusters. The graph showed a noticeable bend at k = 4 indicating that increasing the number of clusters beyond this point resulted in only marginal improvements. Therefore, four clusters were selected for the K-Means model")
    
    kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_cluster = df.dropna(
    subset=["Annual_Income", "Spending_Score"]).copy()
    df_cluster["Cluster"] = clusters
    centroids = scaler.inverse_transform(
    kmeans.cluster_centers_)
    
    st.subheader("Customer Segments")
    fig, ax = plt.subplots(figsize=(10,7))
    sns.scatterplot(
    data=df_cluster,
    x="Annual_Income",
    y="Spending_Score",
    hue="Cluster",
    palette="Set2",
    s=70,
    ax=ax)
    ax.scatter(
        centroids[:,0],
        centroids[:,1],
        c="black",
        s=250,
        marker="X",
        label="Centroids")
    ax.set_title("Customer Segments using K-Means")
    st.pyplot(fig)
    st.write("The clustered scatter plot illustrates the customer segments identified by the K-Means algorithm. Each colour represents a unique customer segment, while the black markers indicate the cluster centroids. The visualization demonstrates that the algorithm successfully separated customers into distinct groups based on annual income and spending behaviour. The centroids represent the average characteristics of each cluster and serve as the reference point for assigning customers to their respective segments.")
    
    st.subheader("📊 Cluster Characteristics")
    cluster_summary = (
        df_cluster.groupby("Cluster")
        .agg({
            "Age": "mean",
            "Annual_Income": "mean",
            "Spending_Score": "mean"
        })
        .round(1)
    )

    cluster_summary["Customers"] = df_cluster.groupby("Cluster").size()

    cluster_summary.columns = [
        "Average Age",
        "Average Income (k$)",
        "Average Spending Score",
        "Customers"
    ]

    st.table(cluster_summary)
    
    cluster_description = pd.DataFrame({
    "Cluster": [
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
        "Cluster 3"
    ],
    "Customer Profile": [
        "Young customers with moderate income and high spending behaviour.",
        "Customers with high income but relatively low spending behaviour.",
        "Customers with lower income and lower spending behaviour.",
        "Customers with moderate to high income and moderate spending behaviour."
    ],
    "Business Strategy": [
        "Retain through loyalty programs, exclusive offers, and premium memberships.",
        "Encourage higher spending using personalized promotions and premium product recommendations.",
        "Increase engagement through discounts, seasonal offers, and value-for-money products.",
        "Maintain customer satisfaction with personalized recommendations and targeted marketing campaigns."
    ]})

    st.table(cluster_description)

    st.subheader("Model Performance")
    st.metric(
    "Silhouette Score",
    "0.395")
    st.write("The model achieved a Silhouette Score of 0.3948, indicating a reasonable level of cluster separation and cohesion. Although the clusters are not perfectly distinct, the score suggests that the K-Means algorithm successfully identified meaningful customer groups within the dataset.")

elif page == "Business Insights":

    st.title("💡 Business Insights")
    
    st.subheader("Project Summary")

    st.write("""
             This project successfully applied K-Means clustering to segment customers based on
             their annual income and spending behaviour. Exploratory Data Analysis identified
             meaningful relationships between customer attributes, while clustering grouped
             customers into distinct segments with similar purchasing characteristics. These
             segments enable businesses to better understand customer behaviour and develop
             targeted marketing strategies.""")
             
    st.subheader("📈 Business Insights")
    business_insights = pd.DataFrame({
    "Observation":[
        "High-spending customers",
        "Low-spending customers",
        "Income is not the only factor",
        "Customer segmentation"
    ],
    "Business Insight":[
        "Reward loyal customers with premium memberships and exclusive offers.",
        "Increase engagement through discounts, seasonal campaigns, and promotional offers.",
        "Customers with higher income do not always spend more, highlighting the importance of behavioural analysis.",
        "Different customer groups require personalized marketing strategies instead of a one-size-fits-all approach."
    ]})

    st.table(business_insights)
    
    st.subheader("🎯 Recommendations")
    st.success("""
    • Personalize marketing campaigns for different customer segments.

    • Offer loyalty rewards to retain high-value customers.

    • Design promotional campaigns for low-spending customer groups.

    • Use customer segmentation to improve product recommendations and inventory planning.

    • Regularly update customer segments as new data becomes available.
    """)
    
    st.subheader("Conclusion")
    st.write("""
    Customer segmentation provides valuable insights into customer purchasing behaviour
    by grouping customers with similar characteristics. The K-Means clustering model
    successfully identified meaningful customer segments that can support data-driven
    decision-making. These insights enable businesses to improve customer engagement,
    increase marketing effectiveness, and allocate resources more efficiently.
    """)
    
    st.subheader("Future Scope")
    st.info("""
    Future improvements could include incorporating additional customer attributes,
    experimenting with advanced clustering algorithms such as DBSCAN or Hierarchical
    Clustering, integrating real-time customer data, and deploying the solution as a
    fully interactive business intelligence application connected to live databases.
    """)






















