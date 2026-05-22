import streamlit as st

pages = st.navigation([
    st.Page("pages/01_Presentation_Dataviz.py", title="Présentation & Dataviz", icon="📊"),
    st.Page("pages/00_Background_MLOps.py",     title="Background MLOps",        icon="🚲"),
    st.Page("pages/02_Validation.py",           title="Tests et Validations",    icon="🏥"),
    st.Page("pages/03_Prediction.py",           title="Prédiction",              icon="🔮"),
    st.Page("pages/04_Prochaines_Etapes.py",    title="Les prochaines étapes",   icon="🚀"),
])
pages.run()
