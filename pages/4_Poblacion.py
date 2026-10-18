import streamlit as st

from mavericks import categories as cat
from mavericks import population_view as pv
from mavericks.ui_common import get_resources, setup

setup("Población", "👥")
st.title("Población sintética")
st.markdown(
    "50 000 personas adultas que representan a la población adulta residente en España. "
    "Se construyen reponderando el Censo 2021 a las cifras oficiales de 1 de enero de 2026."
)

resources = get_resources()
frame = resources.population.frame
targets = pv.load_targets()
order = {"sexo": cat.SEXO, "grupo_edad": cat.GRUPOS_EDAD, "ccaa": cat.CCAA}

summary = pv.summary(frame, targets)
c1, c2, c3 = st.columns(3)
c1.metric("Personas simuladas", f"{len(frame):,}".replace(",", "."))
c2.metric("Población adulta representada", f"{frame.peso.sum() / 1e6:.1f} M")
if summary:
    c3.metric("Desviación media frente al INE", f"{summary['mean_pp']:.2f} puntos")
    st.caption(f"Desviación máxima en una categoría: {summary['max_pp']:.2f} puntos porcentuales.")
else:
    st.info("Falta el archivo de objetivos del INE: ejecuta `mavericks population-targets`.")

for dimension, label, source in pv.DIMENSIONS:
    st.subheader(label)
    table = pv.margin_table(frame, dimension, order[dimension], (targets or {}).get(dimension))
    shown = table.rename(columns={"muestra": "Población sintética (%)", "objetivo": f"{source} (%)", "diferencia": "Diferencia (puntos)"})
    left, right = st.columns([3, 2])
    left.bar_chart(table[[c for c in ("muestra", "objetivo") if c in table]])
    right.dataframe(shown.round(2), use_container_width=True)

st.subheader("Otras variables")
for dimension, label in (("nivel_estudios", "Nivel de estudios"), ("ingresos_hogar_tramo", "Ingresos del hogar")):
    if dimension in frame:
        shares = frame.groupby(dimension).peso.sum()
        st.bar_chart((shares / shares.sum() * 100).rename(label))
st.caption("Estudios e ingresos se ajustan a la EPA y a la Encuesta de Condiciones de Vida del INE.")
