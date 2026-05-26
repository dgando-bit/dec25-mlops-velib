import streamlit as st

pages = st.navigation([
    st.Page("views/01_Presentation_Dataviz.py", title="Présentation & Dataviz", icon="📊"),
    st.Page("views/00_Background_MLOps.py",     title="Architecture MLOps",        icon="🚲"),
    st.Page("views/02_Validation.py",           title="Tests et Validations",    icon="🏥"),
    st.Page("views/03_Prediction.py",           title="Prédiction",              icon="🔮"),
    st.Page("views/04_Prochaines_Etapes.py",    title="Les prochaines étapes",   icon="🚀"),
])
pages.run()
