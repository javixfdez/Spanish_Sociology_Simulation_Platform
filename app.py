import streamlit as st

from mavericks.ui_common import configure, preload

configure()
preload()
navigation = st.navigation(
    [
        st.Page("views/simular.py", title="Simular", icon=":material/analytics:", default=True),
        st.Page("views/lote.py", title="Lote", icon=":material/folder_copy:"),
        st.Page("views/comparar.py", title="Comparar", icon=":material/compare_arrows:"),
        st.Page("views/validacion.py", title="Validación", icon=":material/verified:"),
        st.Page("views/metodologia.py", title="Metodología", icon=":material/account_tree:"),
        st.Page("views/poblacion.py", title="Población", icon=":material/groups:"),
    ]
)
navigation.run()
