from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Resultado académico",
    page_icon="🎓",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

@st.cache_resource
def cargar_recursos():
    paquete = joblib.load(BASE / "modelo_educacion.joblib")
    with open(
        BASE / "metadatos_modelo.json",
        encoding="utf-8"
    ) as archivo:
        metadatos = json.load(archivo)
    return paquete, metadatos

paquete, metadatos = cargar_recursos()
modelo = paquete["modelo"]
variables = paquete["variables"]

traducciones = {
    "Dropout": "Abandono",
    "Enrolled": "Matriculado",
    "Graduate": "Graduado"
}

st.title("🎓 Estimación del resultado académico")
st.caption(
    "ACA 1 · Aprendizaje Automático · CUN · "
    "Wiliam Jhovany Sigindioy Miticanoy"
)

st.write(
    "Prototipo académico que utiliza información del primer semestre "
    "para estimar una de las tres categorías del dataset: "
    "abandono, matriculado o graduado."
)

st.info(
    "El modelo se entrenó con datos de una institución de educación "
    "superior de Portugal. Su desempeño no está validado para la CUN. "
    "La predicción no determina el futuro de un estudiante."
)

with st.form("formulario_academico"):
    st.subheader("Información del estudiante")
    izquierda, derecha = st.columns(2)

    with izquierda:
        edad = st.number_input(
            "Edad al ingresar",
            min_value=17, max_value=70, value=20
        )
        admision = st.number_input(
            "Calificación de admisión · escala 0–200",
            min_value=95.0, max_value=200.0,
            value=127.0, step=0.1
        )
        previa = st.number_input(
            "Calificación de estudios previos · escala 0–200",
            min_value=95.0, max_value=200.0,
            value=133.0, step=0.1
        )
        beca = st.selectbox(
            "¿Tiene beca?",
            options=["No", "Sí"]
        )

    with derecha:
        inscritas = st.number_input(
            "Asignaturas inscritas en el primer semestre",
            min_value=0, max_value=26, value=6
        )
        evaluaciones = st.number_input(
            "Número de evaluaciones en el primer semestre",
            min_value=0, max_value=45, value=8
        )
        aprobadas = st.number_input(
            "Asignaturas aprobadas en el primer semestre",
            min_value=0, max_value=26, value=5
        )
        nota = st.number_input(
            "Calificación del primer semestre · escala 0–20",
            min_value=0.0, max_value=20.0,
            value=12.0, step=0.1
        )

    st.caption(
        "Introduce las calificaciones en las escalas originales "
        "del dataset; no uses directamente notas colombianas de 0 a 5."
    )

    enviado = st.form_submit_button("Estimar resultado")

if enviado:
    if aprobadas > inscritas:
        st.error(
            "Las asignaturas aprobadas no pueden superar "
            "las asignaturas inscritas."
        )
    else:
        datos = {
            "Age at enrollment": edad,
            "Admission grade": admision,
            "Previous qualification (grade)": previa,
            "Scholarship holder": int(beca == "Sí"),
            "Curricular units 1st sem (enrolled)": inscritas,
            "Curricular units 1st sem (evaluations)": evaluaciones,
            "Curricular units 1st sem (approved)": aprobadas,
            "Curricular units 1st sem (grade)": nota
        }

        entrada = pd.DataFrame([datos])[variables]
        prediccion = modelo.predict(entrada)[0]
        probabilidades = modelo.predict_proba(entrada)[0]

        st.subheader(
            f"Resultado estimado: {traducciones[prediccion]}"
        )

        tabla = pd.DataFrame({
            "Categoría": [
                traducciones[clase] for clase in modelo.classes_
            ],
            "Probabilidad estimada": probabilidades
        }).set_index("Categoría")

        st.bar_chart(tabla)
        st.caption(
            "Las probabilidades del modelo no fueron calibradas. "
            "No representan certezas sobre un caso individual."
        )

        with st.expander("Ver valores utilizados"):
            st.dataframe(entrada)

st.divider()
st.subheader("Evaluación del modelo")

metricas = metadatos["metricas_prueba"]
col1, col2, col3 = st.columns(3)

col1.metric("Exactitud en prueba", f"{metricas['accuracy']:.1%}")
col2.metric("F1 macro en prueba", f"{metricas['f1_macro']:.4f}")
col3.metric(
    "Exactitud balanceada",
    f"{metricas['balanced_accuracy']:.4f}"
)

st.caption(
    "Evaluación sobre 885 registros reservados. "
    "La clase Matriculado obtuvo el menor F1: 0,3785."
)

with st.expander("Metodología y fuente"):
    st.write(
        "Se compararon un modelo básico, regresión logística, "
        "árbol de decisión y bosque aleatorio mediante validación "
        "cruzada estratificada de cinco particiones. "
        "El bosque aleatorio se ajustó usando F1 macro."
    )
    st.write(
        "El conjunto de prueba no se utilizó para seleccionar "
        "los parámetros. Se emplearon ocho variables disponibles "
        "al finalizar el primer semestre."
    )
    st.markdown(
        "[Dataset original en UCI]"
        "(https://archive.ics.uci.edu/dataset/697/"
        "predict+students+dropout+and+academic+success)"
    )
