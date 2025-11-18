import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load Data
attendance = pd.read_csv("attendance_logs.csv")
events = pd.read_csv("event_participation.csv")
lms = pd.read_csv("lms_usage.csv")

st.title("📊 Smart Campus Insights")
st.sidebar.header("Filters")

# Student Filter
students = attendance['StudentID'].unique()
selected = st.sidebar.multiselect("Select Students", students, default=students)

# Filtered Data
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
st.subheader("💻 LMS Usage")
lms_summary = fl.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
st.dataframe(lms_summary)

# ML Model
st.subheader("🤖 Engagement Prediction")

ml_att = attendance.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate')
ml_lms = lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index()

ml_data = pd.merge(ml_att, ml_lms, on='StudentID')
ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
y = ml_data['Engagement']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = DecisionTreeClassifier().fit(X_train, y_train)

st.text("Model Performance:")
st.text(classification_report(y_test, model.predict(X_test)))

# New Student Prediction
st.subheader("📈 Predict New Student Engagement")

ar = st.number_input("Absence Rate", 0.0, 1.0, 0.1)
sd = st.number_input("Session Duration (min)", 0.0, 30.0)
pv = st.number_input("Pages Viewed", 0.0, 10.0)

if st.button("Predict"):
    pred = model.predict([[ar, sd, pv]])[0]
    st.success("Engaged" if pred == 1 else "At Risk")
