import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(layout="wide")

# ==============================
# Load Data
# ==============================
attendance_df = pd.read_csv("attendance_logs.csv")
events_df = pd.read_csv("event_participation.csv")
lms_df = pd.read_csv("lms_usage.csv")

attendance_df['Date'] = pd.to_datetime(attendance_df['Date'])

# ==============================
# Sidebar Filters
# ==============================
st.sidebar.header("🔍 Filters")

# Student selection
students = attendance_df['StudentID'].unique()
selected_students = st.sidebar.multiselect("Select Students", students, default=students)

# Date filter
date_range = st.sidebar.date_input("Select Date Range", [])
if len(date_range) == 2:
    start, end = date_range
    attendance_df = attendance_df[(attendance_df["Date"] >= pd.to_datetime(start)) &
                                  (attendance_df["Date"] <= pd.to_datetime(end))]

# Event type filter
if "EventType" in events_df.columns:
    event_types = events_df["EventType"].unique()
    selected_event_type = st.sidebar.multiselect("Event Type", event_types, default=event_types)
    events_df = events_df[events_df["EventType"].isin(selected_event_type)]

# Filter datasets
filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]


# ==============================
# Dashboard KPIs
# ==============================
st.title("📊 Smart Campus Insights")

col1, col2, col3, col4 = st.columns(4)

attendance_percent = (filtered_attendance['Status'].eq("Present").mean() * 100)
avg_session_duration = filtered_lms['SessionDuration'].mean()
avg_pages = filtered_lms["PagesViewed"].mean()
total_events = filtered_events.shape[0]

col1.metric("Attendance %", f"{attendance_percent:.1f}%")
col2.metric("Avg Session Duration", f"{avg_session_duration:.1f} mins")
col3.metric("Avg Pages Viewed", f"{avg_pages:.1f}")
col4.metric("Total Events Attended", total_events)


# ==============================
# Attendance Trends
# ==============================
st.subheader("📋 Attendance Trends")

attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
st.line_chart(attendance_summary)


# ==============================
# EXTRA ANALYTICS — New Data Points
# ==============================

st.subheader("📌 Additional Attendance Insights")

# Attendance percentage per student
att_pct = (filtered_attendance['Status'].eq("Present")
           .groupby(filtered_attendance['StudentID'])
           .mean() * 100).reset_index(name="Attendance%")

# Consecutive absences
def longest_absence_streak(df):
    df = df.sort_values("Date")
    streak = max_streak = 0
    for status in df["Status"]:
        if status == "Absent":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

absence_streak = filtered_attendance.groupby("StudentID").apply(longest_absence_streak)
absence_streak = absence_streak.reset_index(name="MaxAbsenceStreak")

att_extra = pd.merge(att_pct, absence_streak, on="StudentID")

st.dataframe(att_extra)


# ==============================
# Event Participation
# ==============================
st.subheader("🎓 Event Participation")
event_counts = filtered_events['EventName'].value_counts()
st.bar_chart(event_counts)



fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(weekly_usage, ax=ax)
st.pyplot(fig)


# ==============================
# Correlation Analysis
# ==============================
st.subheader("📈 Correlation Between Metrics")

corr_data = ml_data = pd.merge(
    attendance_df.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate'),
    lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index(),
    on='StudentID'
)

corr = corr_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)


# ==============================
# ML Model — Expanded Features
# ==============================
st.subheader("🤖 Predict Student Engagement Risk")

ml_data['Consistency'] = lms_df.groupby("StudentID")['SessionDuration'].std().reset_index(drop=True)

ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed', 'Consistency']]
y = ml_data['Engagement']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Model Performance:")
st.text(classification_report(y_test, y_pred))


# ==============================
# Predict New Student
# ==============================
st.subheader("📈 Predict Engagement for New Student")

absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
session_duration = st.number_input("Avg Session Duration (minutes)", min_value=0.0, value=30.0)
pages_viewed = st.number_input("Avg Pages Viewed", min_value=0.0, value=10.0)
consistency = st.number_input("Session Duration Std Dev (Consistency Score)", min_value=0.0, value=5.0)

if st.button("Predict Engagement"):
    prediction = model.predict([[absence_rate, session_duration, pages_viewed, consistency]])
    result = "Engaged" if prediction[0] == 1 else "At Risk"
    st.success(f"Predicted Engagement Status: {result}")
