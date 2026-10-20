import streamlit as st

from mavericks.ui_common import configure, preload

configure()
preload()
navigation = st.navigation(
    [
        st.Page("views/simular.py", title="Simular", icon="📊", default=True),
        st.Page("views/lote.py", title="Lote", icon="🗂️"),
        st.Page("views/validacion.py", title="Validación", icon="✅"),
        st.Page("views/metodologia.py", title="Metodología", icon="🧭"),
        st.Page("views/poblacion.py", title="Población", icon="👥"),
    ]
)
navigation.run()
