# import necessary libraries
import streamlit as st
import joblib
import pandas as pd
import time


# page configuration
# page tab naming and icon
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🫂",
    layout="wide",  # use wide layout for the two-part design
    initial_sidebar_state="collapsed",
)


# load model
@st.cache_resource
def load_model():
    """Load the trained model from file."""
    try:
        model = joblib.load("tuned_random_forest_model.pkl")
        return model
    except FileNotFoundError:
        return None

model = load_model()


if model is None:
    st.error("Fatal Error: Model file 'tuned_random_forest_model.pkl' not found.")
    st.info("Please ensure the model file is in the same directory as this script.")
    st.stop()


# using css for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    /* general body styling */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("https://images.unsplash.com/photo-1558537112-ed307f81124c?q=80&w=1770&auto=format&fit=crop&ixlib-rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"); /* background image with overlay to not make the background image be the main focus */
        background-size: cover;
        background-position: center;
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }

            
    /* main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

            
    /* title and header Styling */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

            
    /* form container with translucent overlay */
    [data-testid="stForm"] {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
    }

            
    /* button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 25px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        display: block;
        margin: 1rem auto 0 auto;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }

            
    /* prediction result styling */
    .prediction-box {
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0; 
        margin-bottom: 1.5rem; 
        border: 4px solid;
    }
    .prediction-positive {
        background-color: rgba(255, 235, 238, 0.9);
        color: #d32f2f;
        border-color: #d32f2f;
    }
    .prediction-negative {
        background-color: rgba(232, 245, 233, 0.9);
        color: #388e3c;
        border-color: #388e3c;
    }
    .disclaimer-text {
        font-size: 0.8rem;
        font-weight: 400;
        opacity: 0.9;
        margin-top: 1rem;
    }
    
            
    /* depression factor card styling */
    .factor-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        border-left: 8px solid;
        height: 100%; /* Make cards in a row equal height */
    }
    .factor-card h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .factor-card p {
        margin-bottom: 0.2rem;
        font-size: 0.9rem;
    }
    .factor-card .impact {
        font-weight: 600;
        font-size: 1rem;
    }
    .high-impact {
        border-left-color: #d32f2f;
        color: #d32f2f;
    }
    .low-impact {
        border-left-color: #388e3c;
        color: #388e3c;
    }
    .neutral-impact {
        border-left-color: #fbc02d;
        color: #333;
    }

            
    /* dynamic info box styling */
    .info-box {
        background-color: rgba(0,0,0,0.4);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 0; /* Adjusted for new layout */
        border-left: 5px solid;
    }
    .info-box h4 {
        margin-top:0;
        margin-bottom: 0.5rem;
    }
    .info-box ul, .info-box ol {
        padding-left: 20px;
        margin-bottom: 0;
    }
    .info-box li {
        margin-bottom: 0.5rem;
    }
    .resources-box {
        border-left-color: #d32f2f;
    }
    .resources-box h4 {
        color: #f8d7da;
    }
    .tips-box {
        border-left-color: #388e3c;
    }
    .tips-box h4 {
        color: #d4edda;
    }

</style>
""", unsafe_allow_html=True)


# page navigation logic 
if 'page' not in st.session_state:
    st.session_state.page = 'main'


def show_main_page():
    # Function to display the main page with introduction and prediction form.
    # main title 
    st.markdown("<h1 style='text-align: center;'>🧘🏻‍♀️ Student Depression Predictor</h1>", unsafe_allow_html=True)


    # two-column layout for the main content
    left_column, right_column = st.columns([1, 1], gap="large")


    # left column: introduction, and disclaimer
    with left_column:
        # about the predictior container
        st.markdown("""
        <div style="background-color: rgba(255, 239, 213, 0.2); border-left: 5px solid #afd5f0; color: #FFFFFF; padding: 1rem; border-radius: 8px; margin-top: 1.5rem; text-align: justify;">
        <h4 style="color: #afd5f0; margin-top:0;">About the predictor</h4>
        <p>This tool is designed to provide students with a confidential and insightful way to reflect on their well-being. By answering a few questions about your lifestyle and academic pressures, our AI model can help identify potential signs of stress and depression.</p>
        <p>Your privacy is our priority. All responses are anonymous and are not stored. Please answer the questions honestly to get the most accurate prediction.</p>
        </div>
        """, unsafe_allow_html=True)

        # important disclaimer container
        st.markdown("""
        <div style="background-color: rgba(255, 239, 213, 0.2); border-left: 5px solid #ffA500; color: #FFFFFF; padding: 1rem; border-radius: 8px; margin-top: 1.5rem; text-align: justify;">
            <h4 style="color: #ffA500; margin-top:0;">Important Disclaimer</h4>
            <p style="color: #FFFFFF;">This prediction is not a clinical diagnosis and is not 100% accurate. It is an informational tool based on patterns from student data. If you are struggling or need support, please seek guidance from a qualified mental health professional or a university counselor.</p>
        </div>
        """, unsafe_allow_html=True)


    # right column: prediction form and button
    with right_column:
        with st.form("prediction_form"):
            st.markdown("<h4 style='text-align: center;'>Please answer the following questions truthfully</h4>", unsafe_allow_html=True)
            
            # help tooltips for each input field - to make the website more user-friendly.
            age = st.slider("Age", 18, 60, 25, help="Please select your current age.")
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True, help="Please select your gender.")
            
            degree_options = [
                'B.Tech', 'B.Com', 'BSc', 'BA', 'BBA', 'BE', 'BCA', 'M.Tech', 'MBA', 
                'MSc', 'MA', 'PhD', 'MBBS', 'LLB', 'B.Ed', 'B.Pharm', 'M.Com', 
                'Class 12', 'ME', 'M.Ed', 'M.Pharm', 'BHM', 'MD', 'LLM', 'MHM', 'Others'
            ]
            degree = st.selectbox("Highest Qualification", degree_options, help="Please select your highest current or completed qualification.")
            
            academic_pressure = st.slider("Academic Pressure", 0, 5, 3, help="Rate your current academic pressure on a scale of 0 to 5, where 0 is no pressure and 5 is very high pressure.")
            study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3, help="Rate your satisfaction with your studies on a scale of 0 to 5, where 0 is not satisfied at all and 5 is very satisfied.")
            financial_stress = st.slider("Financial Stress", 1, 5, 3, help="Rate your level of financial stress on a scale of 1 to 5, where 1 is no stress and 5 is very high stress.")
            
            family_history = st.radio("Family History of Mental Illness", ["Yes", "No"], horizontal=True, help="Has anyone in your family been diagnosed with a mental illness?")
            suicidal_thoughts = st.radio("History of Suicidal Thoughts", ["Yes", "No"], horizontal=True, help="Have you ever experienced suicidal thoughts?")
            
            sleep_duration = st.selectbox("Average Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"], help="On average, how many hours of sleep do you get per night?")
            dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"], help="How would you describe your typical diet?")
            work_study_hours = st.slider("Work/Study Hours per Day", 0, 12, 6, help="On average, how many hours do you spend working or studying each day?")
            

            submitted = st.form_submit_button("Predict My Depression Risk")

            if submitted:
                with st.spinner("Analysing your inputs..."):
                    time.sleep(2) # simulate processing time using loading spinner

                    # data Preparation
                    model_features = model.feature_names_in_
                    input_data = {feature: 0 for feature in model_features}
                    input_data.update({
                        'Age': age, 'Academic Pressure': float(academic_pressure),
                        'Study Satisfaction': float(study_satisfaction), 'Financial Stress': float(financial_stress),
                        'Work/Study Hours': float(work_study_hours), 'Work Pressure': 0.0,
                        'CGPA': 0.0, 'Job Satisfaction': 0.0
                    })
                    if gender == 'Male': input_data['Gender_Male'] = 1
                    if f"Sleep Duration_{sleep_duration}" in input_data: input_data[f"Sleep Duration_{sleep_duration}"] = 1
                    if f"Dietary Habits_{dietary_habits}" in input_data: input_data[f"Dietary Habits_{dietary_habits}"] = 1
                    if suicidal_thoughts == 'Yes': input_data['Have you ever had suicidal thoughts ?_Yes'] = 1
                    if family_history == 'Yes': input_data['Family History of Mental Illness_Yes'] = 1
                    # create the column name based on user's selection
                    degree_column_name = f"Degree_{degree}"
                    # check if this column exists in the model's features
                    if degree_column_name in input_data:
                        input_data[degree_column_name] = 1
                    else:
                        # if user selected 'Others', default to 'Others'
                        input_data['Degree_Others'] = 1
                    input_df = pd.DataFrame([input_data])[model_features]

                    # prediction
                    st.session_state.prediction = model.predict(input_df)[0]
                    st.session_state.prediction_proba = model.predict_proba(input_df)
                    
                    # store original inputs for display
                    st.session_state.inputs = {
                        "Suicidal Thoughts": {"value": suicidal_thoughts, "is_risk": suicidal_thoughts == "Yes", "impact_text": "High Impact"},
                        "Academic Pressure": {"value": f"{academic_pressure}/5", "is_risk": academic_pressure > 3, "impact_text": "High Impact" if academic_pressure > 3 else ("Low Impact" if academic_pressure < 2 else "Neutral Impact")},
                        "Family History": {"value": family_history, "is_risk": family_history == "Yes", "impact_text": "High Impact"},
                        "Financial Stress": {"value": f"{financial_stress}/5", "is_risk": financial_stress > 3, "impact_text": "High Impact" if financial_stress > 3 else ("Low Impact" if financial_stress < 2 else "Neutral Impact")},
                        "Sleep Duration": {"value": sleep_duration, "is_risk": sleep_duration in ["Less than 5 hours", "5-6 hours"], "impact_text": "High Impact" if sleep_duration in ["Less than 5 hours", "5-6 hours"] else "Low Impact"},
                        "Study Satisfaction": {"value": f"{study_satisfaction}/5", "is_risk": study_satisfaction < 2, "impact_text": "High Impact" if study_satisfaction < 2 else ("Low Impact" if study_satisfaction > 3 else "Neutral Impact")}
                    }
                    # switch to the results page
                    st.session_state.page = 'results'
                    st.rerun()


def show_results_page():
    # Function to display the results page with prediction, tips, and analysis.
    # two-column layout for the detailed results
    left_column, right_column = st.columns([1, 1.5], gap="large")

    # left column: prediction, dynamic tips, and predict again button
    with left_column:
        # main prediction result container

        # if prediction is positive (high risk of depression), display a red box with the prediction result and probability.
        if st.session_state.prediction == 1:
            st.markdown(f"""
            <div class="prediction-box prediction-positive">
                <h2 style='text-align: center; color: #d32f2f;'>Your Predicted Results</h2>
                <p>Prediction: High risk of Depression ({st.session_state.prediction_proba[0][1]*100:.2f}% probability)</p>
                <p class="disclaimer-text">This is not a diagnosis. Should you require help, please consult a professional for guidance.</p>
            </div>""", unsafe_allow_html=True)


        # else, if prediction is negative (low risk of depression), display a green box with the prediction result and probability.
        else:
            st.markdown(f"""
            <div class="prediction-box prediction-negative">
                <h2 style='text-align: center; color: #388e3c;'>Your Predicted Results</h2>
                <p>Prediction: Low risk of Depression ({st.session_state.prediction_proba[0][0]*100:.2f}% probability)</p>
                <p class="disclaimer-text">This is not a diagnosis. Continue to prioritise your well-being.</p>
            </div>""", unsafe_allow_html=True)


        # dynamic tips container (content changes based on prediction result).
        # if result is positive (high risk of depression), display depression next steps and resources.
        if st.session_state.prediction == 1:
            st.markdown("""
            <div class="info-box resources-box">
                <h4>Next Steps & Resources</h4>
                <p>It's important to address these feelings. Consider taking these steps:</p>
                <ul>
                    <li><b>Talk to Someone:</b> Reach out to a friend, family member, or university counselor.</li>
                    <li><b>Seek Professional Help:</b> A mental health professional can provide guidance and support.</li>
                    <li><b>University Resources:</b> Check your university's wellness center for available services.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


        # else, if result is negative (low risk of depression), display tips for maintaining well-being.
        else:
            st.markdown("""
            <div class="info-box tips-box">
                <h4>Maintaining Your Well-being</h4>
                <p>It's great that you're in a positive space. Here are some tips to maintain it:</p>
                <ol>
                    <li><b>Stay Connected:</b> Continue to nurture your social connections.</li>
                    <li><b>Mindful Habits:</b> Keep up with healthy sleep and dietary patterns.</li>
                    <li><b>Manage Stress:</b> Proactively manage academic and financial stress with planning and support.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        

        # predict again button
        if st.button("Predict Again"):
            st.session_state.page = 'main'
            # clear previous results to ensure a fresh start
            for key in ['prediction', 'prediction_proba', 'inputs']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    

    # right column: depression factor card analysis cards display
    with right_column:
        # display the depression factor card analysis that is based on the user's input and the model's prediction (result)
        st.markdown("<h3 style='text-align: center;'>Your Personalised Results Analysis</h3>", unsafe_allow_html=True)
        
        cols = st.columns(2) # split the display into two columns
        col_idx = 0
        for factor, data in st.session_state.inputs.items():

            # determine the impact class based on the risk and impact text
            impact_class = "high-impact" if data["is_risk"] else ("low-impact" if data["impact_text"] == "Low Impact" else "neutral-impact")

            # this logic is for factors where high values are good (satisfaction)
            if data["impact_text"] == "Low Impact" and not data["is_risk"]:
                impact_class = "low-impact"
            elif data["impact_text"] == "High Impact" and not data["is_risk"]:
                impact_class = "low-impact"
            

            card_html = f"""
            <div class="factor-card {impact_class}">
                <h4>{factor}</h4>
                <p>Your Input: <b>{data['value']}</b></p>
                <p class="impact">{data['impact_text']}</p>
            </div>
            """

            # display the card in the appropriate column
            with cols[col_idx]:
                st.markdown(card_html, unsafe_allow_html=True)
            col_idx = (col_idx + 1) % 2

# app router - to route between the main page and results page based on session state.
# checks which page to show.
if st.session_state.page == 'main':
    show_main_page()
elif st.session_state.page == 'results':
    show_results_page()