# Telecom User Analysis and Dashboard Development

## Project Description

This project analyzes a telecommunication dataset to evaluate the growth potential of TellCo, a mobile service provider in the Republic of Pefkakia. By understanding user behavior, engagement, experience, and satisfaction, the project aims to provide actionable insights and recommendations to an investor considering acquiring TellCo. 

The outcomes include an interactive dashboard for visualizing key findings, a detailed report, and predictive models for customer satisfaction.

## Objectives

1. **Understand Customer Behavior:** Analyze handset preferences, application usage, and session details.
2. **Engagement Patterns:** Identify metrics such as session count, duration, and data traffic, and classify users into engagement levels.
3. **User Experience Analysis:** Investigate network performance metrics (TCP retransmission, RTT, throughput) and their impact on user experience.
4. **Satisfaction Evaluation:** Calculate satisfaction scores based on engagement and experience metrics and predict satisfaction using machine learning.
5. **Dashboard Development:** Build an interactive Streamlit dashboard for presenting findings to stakeholders.

## Proposed Solution

### Tools and Technologies
- **Python:** For data processing, analysis, and model building.
- **PostgreSQL:** To extract and manage the telecommunication dataset.
- **Streamlit:** For creating the interactive dashboard.
- **Scikit-learn:** For clustering and predictive modeling.
- **Matplotlib/Seaborn/Plotly:** For data visualization.
- **Docker or MLflow:** To deploy and track machine learning models.
- **GitHub Actions:** For CI/CD pipelines to automate testing and deployment.

### Procedure i followed 

#### 1. **Data Preparation**
   - Connect to the PostgreSQL database and extract the dataset.
   - Clean the data by handling missing values, outliers, and duplicates.
   - Transform and normalize variables for analysis.

#### 2. **Exploratory Data Analysis (EDA)**
   - Conduct User Overview Analysis to understand handset preferences and user behavior.
   - Visualize data using plots to uncover trends and insights.

#### 3. **Engagement and Experience Analysis**
   - Aggregate engagement metrics (e.g., session counts, data usage).
   - Use clustering to classify users into engagement and experience groups.
   - Analyze network performance metrics (e.g., RTT, throughput) by handset type.

#### 4. **Satisfaction Analysis**
   - Calculate engagement, experience, and satisfaction scores.
   - Build a regression model to predict customer satisfaction.

#### 5. **Dashboard Development**
   - Create an interactive dashboard with tabs for each analysis section.
   - Deploy the dashboard to Heroku or a similar platform for public access.

#### 6. **Documentation and Deployment**
   - Write a comprehensive report summarizing the findings and recommendations.
   - Organize the project repository with modular Python scripts, unit tests, and CI/CD pipelines.
   - Publish the code and deliverables on GitHub.

## Folder Structure
```plaintext
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows
│       ├── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md
```

## Deliverables
1. An interactive dashboard accessible via a public URL.
2. A detailed report with actionable recommendations.
3. Reusable Python scripts for data processing, analysis, and modeling.
4. A GitHub repository with organized code, unit tests, and CI/CD pipelines.
## Author
Yonatan Abrham
- Email: [email2yonatan@gmail.com](mailto:email2yonatan@gmail.com)
- LinkedIn: [Yonatan Abrham](https://www.linkedin.com/in/yonatan-abrham1/)
- GitHub: [YonInsights](https://github.com/YonInsights)
