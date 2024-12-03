import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# App title
st.set_page_config(page_title="Credit Card Fraud Detection App", layout="wide")
st.title("🔍 Credit Card Fraud Detection App")

# Sidebar for file uploads
st.sidebar.header("Data Input")
uploaded_files = st.sidebar.file_uploader(
    "Upload your CSV files", type=["csv"], accept_multiple_files=True
)

# Initialize containers for datasets
datasets = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        data = pd.read_csv(uploaded_file)
        # Normalize column names to handle inconsistencies
        data.columns = data.columns.str.strip().str.lower()
        datasets[uploaded_file.name] = data
    st.success(f"Uploaded {len(uploaded_files)} datasets successfully!")

if datasets:
    for name, data in datasets.items():
        st.write(f"## Dataset: {name}")

        # Display dataset
        with st.container():
            st.write("### Preview of the Dataset")
            st.dataframe(data.head(), use_container_width=True)

        # Data summary
        st.write("### Dataset Summary")
        st.write(data.describe())

        # Visualization options
        st.sidebar.header(f"Visualization for {name}")

        # Correlation Heatmap
        if st.sidebar.checkbox(f"Show Correlation Heatmap ({name})"):
            st.write("### Correlation Heatmap")
            # Increased figure size for clarity
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                data.corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar=True,
                ax=ax,
                annot_kws={"size": 10}  # Adjusted font size for annotations
            )
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            st.pyplot(fig)

        # Pie chart for class distribution
        if 'class' in data.columns:
            st.write("### Class Distribution")
            class_counts = data['class'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(class_counts, labels=class_counts.index,
                   autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
            # Equal aspect ratio ensures pie chart is a circle
            ax.axis('equal')
            st.pyplot(fig)

            # Model training and evaluation
            X = data.drop('class', axis=1)
            y = data['class']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Sidebar for model parameters
            st.sidebar.header(f"Model Configuration ({name})")
            n_estimators = st.sidebar.slider(
                f"Number of Trees ({name})", 10, 200, 100)
            max_depth = st.sidebar.slider(f"Maximum Depth ({name})", 5, 50, 10)

            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Display model performance
            with st.container():
                st.write("### Model Performance")
                st.metric("Accuracy", f"{accuracy:.2f}")
                st.write("#### Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
                            "Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"], ax=ax)
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                st.pyplot(fig)
                st.write("#### Classification Report")
                st.text(classification_report(y_test, y_pred))

            # Feature Importances
            st.write("### Feature Importances")
            feature_importances = pd.DataFrame(
                {'Feature': X.columns, 'Importance': model.feature_importances_}
            )
            feature_importances = feature_importances.sort_values(
                by='Importance', ascending=False
            )
            st.bar_chart(feature_importances.set_index('Feature'))

            # Download results
            st.sidebar.header(f"Download Results ({name})")
            if st.sidebar.button(f"Download Feature Importances ({name})"):
                csv = feature_importances.to_csv(index=False)
                st.sidebar.download_button(
                    f"Download CSV ({name})", csv, f"feature_importances_{name}.csv", "text/csv"
                )
        else:
            st.warning(
                f"Dataset {name} does not have a 'class' column. Adding a placeholder column for testing.")
            data['class'] = np.random.randint(0, 2, size=data.shape[0])

            # Notify the user
            st.info(
                f"A placeholder 'class' column has been added to dataset {name}. Please ensure the real column exists for actual use.")
else:
    st.info("Upload datasets to get started.")

# Footer
st.sidebar.write("---")
st.sidebar.write("🔧 Developed with ❤️ using Streamlit")
