import streamlit as st
import pandas as pd

# Load Data
attendance = pd.read_csv("attendance_logs.csv")
events = pd.read_csv("event_participation.csv")
lms = pd.read_csv("lms_usage.csv")

st.title("📊 Smart Campus Insights")
st.sidebar.header("Filters")

# Student Filter
students = attendance['StudentID'].unique()
selected = st.sidebar.multiselect("Select Students", students, default=students)

# Filter Data
fa = attendance[attendance['StudentID'].isin(selected)]
fe = events[events['StudentID'].isin(selected)]
fl = lms[lms['StudentID'].isin(selected)]

# Attendance Chart
st.subheader("📋 Attendance Trends")
att_summary = fa.groupby(['Date', 'Status']).size().unstack(fill_value=0)
st.line_chart(att_summary)

# Event Participation
st.subheader("🎓 Event Participation")
st.bar_chart(fe['EventName'].value_counts())

# LMS Usage
st.subheader("💻 LMS Usage Patterns")
lms_summary = fl.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
st.dataframe(lms_summary)
