import pandas as pd
import streamlit as st
import altair as alt
import numpy as np 
import joblib
alt.data_transformers.disable_max_rows()
st.set_page_config(page_title = "Maintenance Predictions", page_icon= "⚙")
# Headings
st.title("Predictive Maintenance for Industrial Devices")
st.write("*Empowering Manufacturing Efficiency through Smart Maintenance Predictions*")


# Sidebar Input Details
st.sidebar.header('Input Details')

def user_input_features():
    type_device = st.sidebar.selectbox('Type',('L','M','H'))
    air_temperature = st.sidebar.slider('Air Temperature (K)', 295.0, 305.0, 300.0)
    process_temperature = st.sidebar.slider('Process Temperature (K)', 305.0, 314.0, 310.0)
    rotational_speed = st.sidebar.slider("Rotational Speed (RPM)", 1168, 2886, 1500)
    torque = st.sidebar.slider("Torque (N-m)", 3.5, 77.0, 40.0)
    tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 253, 108)
    data = {'Type': type_device,
            'Air Temperature': air_temperature,
            'Process Temperature': process_temperature,
            'Rotational Speed': rotational_speed,
            'Torque': torque,
            'Tool wear': tool_wear
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()
input_df_copy = input_df.copy()

tab1, tab2, tab3 = st.tabs(['Maintenance Prediction', 'Result Explanation', 'About the Project'])
# Tab 1
with tab1:
    st.info('Adjust the sliders or select values in the **sidebar** to input essential operational data', icon="ℹ️")
    st.subheader('Input Details')
    st.write(f"""
            * **Type:** {input_df['Type'].values[0]}
            * **Air Temperature:** {input_df['Air Temperature'].values[0]} K
            * **Process Temperature:** {input_df['Process Temperature'].values[0]} K
            * **Rotational Speed:** {input_df['Rotational Speed'].values[0]} RPM
            * **Torque:** {input_df['Torque'].values[0]} Nm
            * **Tool Wear:** {input_df['Tool wear'].values[0]} min
            """)

# Feature Engineering
input_df['Power'] = 2 * np.pi * input_df['Rotational Speed'] * input_df['Torque'] / 60
input_df['temp_diff'] = input_df['Process Temperature'] - input_df['Air Temperature']
input_df['Type_H'] = 0
input_df['Type_L'] = 0
input_df['Type_M'] = 0
if input_df['Type'].values == 'L':
    input_df['Type_L'] = 1
elif input_df['Type'].values == 'M':
    input_df['Type_M'] = 1
else:
    input_df['Type_H'] = 1

input_df = input_df.drop(['Type', 'Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque'], axis = 1)

model = joblib.load("predictive_maintenance.pkl")
prediction = model.predict(input_df)
prediction_probability = model.predict_proba(input_df)

# Tab 1
with tab1:
    st.subheader('Prediction')
    if prediction == 0:
        st.success('No Maintenance Required', icon="✅")
        st.write(f"Probability: **{list(prediction_probability)[0][0]:.2%}**")
    else:
        st.error('Maintenance Needed', icon="🚨")
        st.write(f"Probability: **{list(prediction_probability)[0][1]:.2%}**")


# Tab 2
with tab2:
    st.write("*See how your input data aligns with the predictive analysis. Understand where your devices stand in comparison to the data, facilitating informed decisions on maintenance priorities.*")
    input_feature = st.selectbox('Feature',('Type', 'Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque', 'Tool Wear'))
    
    data = pd.read_csv("predictive_maintenance.csv")
    data.columns = ['UDI', 'Product ID', 'Type', 'Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque', 'Tool wear', 'Machine failure', 'Failure type']
    data = data.drop(['UDI', 'Product ID', 'Failure type'], axis = 1)
    data = data[data['Machine failure'] == prediction[0]]

    if input_feature == 'Type':
        chart = alt.Chart(data).mark_bar().encode(
            x='Type:O',
            y="count()",
            color=alt.condition(
                alt.datum.Type == input_df_copy['Type'].values[0],
                alt.value('#ff4c4c'), 
                alt.value('steelblue'))
        )
        st.altair_chart(chart, use_container_width = True)
    elif input_feature == 'Air Temperature':
        base = alt.Chart(data)

        bar = base.mark_bar().encode(
            alt.X('Air Temperature:Q').bin().axis(None),
            y='count()',
            color = alt.value('steelblue')
        )
        rule = base.mark_rule(color='#ff4c4c').encode(
            x = alt.datum(input_df_copy['Air Temperature'].values[0]),
            size=alt.value(3)
        )
        st.altair_chart(bar + rule , use_container_width = True)
    elif input_feature == 'Process Temperature':
        base = alt.Chart(data)

        bar = base.mark_bar().encode(
            alt.X('Process Temperature:Q').bin().axis(None),
            y='count()',
            color = alt.value('steelblue')
        )
        rule = base.mark_rule(color='#ff4c4c').encode(
            x = alt.datum(input_df_copy['Process Temperature'].values[0]),
            size=alt.value(3)
        )
        st.altair_chart(bar + rule , use_container_width = True)   
    elif input_feature == 'Rotational Speed':
        base = alt.Chart(data)

        bar = base.mark_bar().encode(
            alt.X('Rotational Speed:Q').bin().axis(None),
            y='count()',
            color = alt.value('steelblue')
        )
        rule = base.mark_rule(color='#ff4c4c').encode(
            x = alt.datum(input_df_copy['Rotational Speed'].values[0]),
            size=alt.value(3)
        )
        st.altair_chart(bar + rule , use_container_width = True) 
    elif input_feature == 'Torque':
        base = alt.Chart(data)

        bar = base.mark_bar().encode(
            alt.X('Torque:Q').bin().axis(None),
            y='count()',
            color = alt.value('steelblue')
        )
        rule = base.mark_rule(color='#ff4c4c').encode(
            x = alt.datum(input_df_copy['Torque'].values[0]),
            size=alt.value(3)
        )
        st.altair_chart(bar + rule , use_container_width = True) 
    else:
        base = alt.Chart(data)

        bar = base.mark_bar().encode(
            alt.X('Tool wear:Q').bin().axis(None),
            y='count()',
            color = alt.value('steelblue')
        )
        rule = base.mark_rule(color='#ff4c4c').encode(
            x = alt.datum(input_df_copy['Tool wear'].values[0]),
            size=alt.value(3)
        )
        st.altair_chart(bar + rule , use_container_width = True) 


# Tab 3
with tab3:
    st.write("""
        Welcome to an innovative project designed to enhance maintenance efficiency for manufacturing companies! Our machine learning model, incorporated into a user-friendly web app, predicts maintenance needs in real time by analyzing data from industrial devices. This proactive solution empowers companies to tackle issues before they cause downtime and increased costs.

        The key components of the project include:

        * **Type:** Products are categorized as low (L), medium (M), or high (H) quality variants, each with a specific serial number.
        * **Air Temperature [K]:** The temperature of the surrounding air, measured in Kelvin.
        * **Process Temperature [K]:** The temperature of the manufacturing process, measured in Kelvin.
        * **Rotational Speed [rpm]:** The speed at which the device rotates, measured in revolutions per minute.
        * **Torque [Nm]:** The applied torque to the device, measured in Newton-meters.
        * **Tool Wear [min]:** The duration of tool usage, measured in minutes.

        To explore the code and understand how our solution automates this process effectively, check out the project on [Kaggle](https://www.kaggle.com/code/atom1991/optimizing-operations-with-predictive-maintenance?kernelSessionId=146948811).
    """)


st.write("---")

st.write("""#### About Me""")
st.write(" ")
st.write("""
            I'm Vivek Agrawal, an accomplished data scientist with expertise in data mining, data visualization, and machine learning.
                 
            [Portfolio](https://www.vivekagrawal.space) | [Linkedin](https://www.linkedin.com/in/ivivekagrawal) | [Github](https://github.com/VivekAgrawl) | [Kaggle](https://www.kaggle.com/atom1991)
        """)

# Disclaimer
st.sidebar.write("---")
st.sidebar.write("""
        _This web app is intended for practical and showcase purposes only. It is part of a project to demonstrate implementation and may not be suitable for critical or production use. The developer assumes no responsibility for any consequences arising from the use of this application._
            
""")