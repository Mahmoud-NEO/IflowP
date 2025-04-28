import streamlit as st
import time

st.set_page_config(
        page_title='IFlowP',
        page_icon='🛎️',
        menu_items={
            'Get help': 'https://www.linkedin.com/in/berat-tu%C4%9Fberk-g%C3%BCnel-928460173/',
            'About': "# Make sure to *cite* while using!"
        },
    )


st.markdown(
    """
    <style>
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        .logo-img {
            width: 120px;
            height: auto;
            margin-right: 10px;
        }
        .logo-text {
            font-size: 20px;
            font-weight: bold;
            color: #FFFFFF;
        }
    </style>
    <div class="logo-container">
        <img class="logo-img" src="https://upload.wikimedia.org/wikipedia/commons/6/69/IFP_Logo.png">
    </div>
    """
    , unsafe_allow_html=True,
)



# Custom CSS for styling 
st.markdown(
    """
    <style>
        /* Sidebar custom label */
        [data-testid="stSidebarNav"]::before {
            content: "IFP School";
            font-family: 'Comic Sans MS', sans-serif;
            margin-left: 100px;
            margin-top: 30px;
            font-size: 21px;
            position: relative;
            top: 5px;
            text-align: center;
            font-weight: bold;
        }
        /* Logo and title styling */
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        .logo-img {
            width: 120px;
            height: auto;
            margin-right: 10px;
        }
        .stTitle {
            font-size: 40px;
            font-weight: bold;
            color: #0256FE;
            margin-bottom: 20px;
            text-align: center;
            font-family: 'Comic Sans MS', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stTitle {
        font-size: 40px;
        font-weight: bold;
        color: #0256FE;
        margin-bottom: 20px;
    }

    .stHeader {
        font-size: 30px;
        font-weight: bold;
        color: #0256FE;
        margin-bottom: 5px;
    
    }

    .stMarkdown {
        font-size: 16px;
        line-height: 1.6;
        color: #ffffff;
    }
    .stTitle, .stHeader, .stMarkdown {
        font-family: 'Comic Sans MS', sans-serif;
    }
    </style>
    """
    , unsafe_allow_html=True
)

st.markdown(
    """
    <div class="stTitle"><center>Gas Well IPR Prediction</center></div>
    """
    , unsafe_allow_html=True
)

with st.sidebar:
    st.spinner("Loading...")
    time.sleep(2)
    st.info("Please choose an *action*")


st.markdown(
    """
    <br>
    <br>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="stMarkdown">
        <strong>This web page is created by the following people:</strong> <br>
        ➔ Berat Tuğberk Günel <br>
        ➔ Mahmoud Mohammed <br>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <br>
    """,
    unsafe_allow_html=True
)

st.info(
    "This project is developed under the supervision of *Daniel Crosby*."
)


st.warning("**This project is created without the expectation of any economic purposes.**", icon="❗")

st.markdown(
    """
    <div class="stMarkdown">
        <br>
        <strong>👈 Please select a prediction method from the sidebar.</strong>
        <br><br>
        <div class="stHeader">What are the constrains?</div>
        <ul>
            <li>During the training, analytical simulation results have been used.</li>
            <li>Since the aim of the project is predicting the IPR curve for gas wells, the skin factor was excluded during the training. But the effect of skin factor was embeded into the training.</li>
            <li>For considerably high flow rates, the prediction could be quite erroneous.</li>
            <li>It is only valid for single phae gas flow.</li>
            <li>Forecasting is based on a single moment in time.</li>
            <li>Reservoir pressure remains constant during depletion.</li>
            <li>Fluid properties are defined at the initial reservoir pressure.</li>
        </ul>
        <br>
        <div class="stHeader">To contact us or the professors:</div>
        <h4>Berat Tuğberk Günel</h4>
        <ul>
            <li>Use LinkedIn: <a href="https://www.linkedin.com/in/berat-tuğberk-günel-928460173/">LinkedIn Profile</a></li>
        </ul>
        <h4>Mahmoud Mohammed</h4>
        <ul>
            <li>Use LinkedIn: <a href="https://www.linkedin.com/in/mahmoud-khaled-petroleum-engineer/">LinkedIn Profile</a></li>
        </ul>
        <h4>Daniel Crosby</h4>
        <ul>
            <li>Use LinkedIn: <a href="https://www.linkedin.com/in/daniel-crosby-09304810/">LinkedIn Profile</a></li>
        </ul>
    </div>
    """
    , unsafe_allow_html=True
)

