# AGENT.md: Project Context & Operational Protocol

## 1. Project Overview & Architecture

This project, Seeing Forecast, uses Machine Learning to predict astronomical "seeing" conditions. The goal is to provide high-accuracy forecasts for astronomical observations based on atmospheric and environmental data.

Documentation Sync: Every time a new file or folder is created, the README.md must be updated immediately with a brief description of its purpose and how it fits into the pipeline.

Repository Map

    /data: Raw and processed datasets.

    /notebooks: Exploratory Data Analysis (EDA) and iterative model prototyping.

    /scripts: Modularized Python scripts for production-grade data pipelines and training.

    /models: Saved model weights and metadata.

    /logs: CRITICAL. Contains session-based error logs and pending tasks.

## 2. Coding Standards & Best Practices
Python Scripts (.py)

    Modularization: Use functions and classes. Avoid monolithic scripts.

    Logic: Minimize heavy if/else nesting. Use dictionaries for mapping or early returns to reduce complexity.

    Documentation: Use Google-style docstrings.

Jupyter Notebooks (.ipynb)

    Illustrative Approach: Do not skip steps (e.g., show data loading, then head, then info).

    Narrative: Write explanatory markdown cells before every code block.

    Simplicity: Prioritize the most readable and maintainable solution over "clever" one-liners.

## 3. Modeling & Evaluation Protocol

You must strictly follow this scientific pipeline for every modeling task:
Data Preparation

    Split Strategy: Always split into Train, Validation, and Test sets.

    Imbalance Check: MANDATORY. Analyze class distribution. If imbalanced, propose solutions (SMOTE, Class Weights, or Resampling) before training.

Iterative Refinement

    Train on Train set.

    Evaluate and tune hyperparameters based on the Validation set.

    Final Step: Only run against the Test set once the model is refined.

Success Metrics

    Priority 1: Best Fit (Generalization/Calibration).

    Priority 2: Best Performance (Accuracy/F1/MSE).

    Visualization: All performance plots must include Standard Deviation and Interquartile Range (25%/75% quartiles).

## 4. Session Memory & Log Management

At the start of every session, you must perform the "Morning Standup":

    Read Logs: Scan the /logs directory for unresolved errors or pending tasks.

    Status Check: Check if these issues were addressed in the last commit/session.

    Offer Solutions: If unresolved, present a new plan to the user. Do not proceed until the user decides to:

        Push: Attempt a new fix.

        Skip/Forget: Abandon the task.

    Log Cleanup: Once a task is solved or skipped, delete the log entry to conserve context memory and keep the workspace clean.

## 5. Error Handling

If you encounter a blocker or a bug you cannot resolve within two attempts:

    Stop coding.

    Write a detailed summary of the error, what was tried, and why it failed into /logs/session_error.log.

    Wait for user intervention.

## 6. Skill-Based Execution & Technical Standards

To ensure scientific rigor and code quality, the agent must consult the technical blueprints located in .cursor/skills/ before starting any task.
Skill Mapping

When performing the following activities, refer to the corresponding skill file for implementation details:

    Data Preprocessing (.cursor/skills/data_preprocessing.md)

        Triggers: Data cleaning, feature engineering, scaling, or handling missing values.

        Key Requirement: Mandatory class imbalance check and proposed solutions.

    Data Visualization (.cursor/skills/data_visualisation.md)

        Triggers: Plotting model results, EDA, or performance metrics.

        Key Requirement: Must include Standard Deviation and IQR (25%/75% quartiles).

    Modeling & Validation (.cursor/skills/modeling.md)

        Triggers: Model selection, training, or evaluation.

        Key Requirement: Strict Train/Val/Test split; prioritize Best Fit over raw performance.

    Deployment (.cursor/skills/deployment_streamlit.md)

        Triggers: Building or updating the Streamlit dashboard.

        Key Requirement: Implement @st.cache and session state management.

Execution Protocol

    Read the Skill: Before writing code for a mapped task, the agent must silently read the relevant .md file in the skills directory.

    Align with SOP: Ensure the code generated aligns with the "Golden Rules" defined in that skill (e.g., if modeling, the agent must automatically include the class imbalance check code).

    Cross-Reference: If a task spans multiple skills (e.g., modeling and plotting results), the agent must synthesize instructions from both.
