"""
Scientific Confusion Matrix Visualization and Metrics Utility for AgriFed-KHAOS.
Adapted from MATLAB plotconfusion style for high-precision agricultural and teledetection diagnostics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger("agrifed_khaos.confusion_matrix")


def plot_pretty_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    out_image_path: str | Path = "confusion_matrix.png",
    cmap: str = "Oranges",
    show_null_values: int = 0,  # 0: celda vacía si es 0, 1: "0", 2: "0 (0%)"
    fz: int = 10,               # Tamaño de fuente base
    title: str = "Matriz de Confusión",
) -> Path:
    """
    Genera y guarda una matriz de confusión de alta resolución con 
    márgenes de precisión de usuario, productor y precisión global (estilo MATLAB plotconfusion).
    
    Dimensiones dinámicas: max(10, N + 1) pulgadas para evitar que las celdas se aplasten.
    """
    out_path = Path(out_image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    else:
        labels = sorted(labels)

    # 1. Calcular matriz básica con sklearn (filas: true, columnas: pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # Guardar también CSV de respaldo
    csv_path = out_path.with_suffix(".csv")
    df_cm.to_csv(csv_path)

    # El eje X representa lo Real ("True") y el eje Y lo Predicho ("Predicted")
    df_cm = df_cm.T

    # 2. Agregar fila y columna de totales
    sum_col = df_cm.sum(axis=0).tolist()
    sum_lin = df_cm.sum(axis=1).tolist()

    df_cm["sum_lin"] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc["sum_col"] = sum_col

    # 3. Dimensionar figura dinámicamente según número de clases (1 pulgada por clase)
    df_len = len(labels) + 1
    fig_dim = max(10, df_len)
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=200)

    # 4. Dibujar heatmap base
    sns.heatmap(
        df_cm,
        annot=True,
        annot_kws={"size": fz},
        linewidths=1.5,
        ax=ax,
        cbar=False,
        cmap=cmap,
        linecolor="black",
        fmt=".2f",
    )

    # Ocultar etiquetas de la fila/columna de totales
    x_ticks = ax.get_xticklabels()
    if x_ticks:
        x_ticks[-1] = ""
    y_ticks = ax.get_yticklabels()
    if y_ticks:
        y_ticks[-1] = ""
    ax.set_xticklabels(x_ticks, fontsize=max(8, fz - 1), rotation=90)
    ax.set_yticklabels(y_ticks, fontsize=max(8, fz - 1), rotation=0)

    # 5. Formatear el contenido de cada celda
    array_df = df_cm.values
    tot_all = array_df[-1, -1]
    n_rows, n_cols = array_df.shape
    text_add = []
    text_del = []

    for t in ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(round(pos[1]))
        col = int(round(pos[0]))
        cell_val = array_df[lin, col]

        t_x, t_y = t.get_position()

        # Celdas marginales de totales (última fila o última columna)
        if col == (n_cols - 1) or lin == (n_rows - 1):
            if cell_val != 0:
                if col == (n_cols - 1) and lin == (n_rows - 1):
                    # Esquina inferior derecha: Precisión Global (Overall Accuracy)
                    tot_rig = sum(array_df[i, i] for i in range(len(labels)))
                    per_ok = (tot_rig / cell_val) * 100
                elif col == (n_cols - 1):
                    # Precisión del Usuario (Fila)
                    tot_rig = array_df[lin, lin]
                    per_ok = (tot_rig / cell_val) * 100
                else:
                    # Precisión del Productor (Columna)
                    tot_rig = array_df[col, col]
                    per_ok = (tot_rig / cell_val) * 100
                per_err = 100.0 - per_ok
            else:
                per_ok = per_err = 0.0

            text_del.append(t)
            font_prop = fm.FontProperties(weight="bold", size=fz)

            # Escribir: conteo, % acierto (verde), % error (rojo)
            base_kw = dict(color="black", ha="center", va="center", fontproperties=font_prop)
            text_add.append(dict(x=t_x, y=t_y - 0.28, text=f"{int(cell_val):,}", kw=base_kw))

            green_kw = base_kw.copy()
            green_kw["color"] = "green"
            text_add.append(dict(x=t_x, y=t_y, text=f"{per_ok:.1f}%", kw=green_kw))

            red_kw = base_kw.copy()
            red_kw["color"] = "red"
            text_add.append(dict(x=t_x, y=t_y + 0.28, text=f"{per_err:.1f}%", kw=red_kw))

        # Celdas internas de la matriz
        else:
            per = (cell_val / tot_all) * 100 if tot_all > 0 else 0
            if cell_val > 0:
                t.set_text(f"$\\mathbf{{{int(cell_val)}}}$\n{per:.2f}%")
            else:
                t.set_text("" if show_null_values == 0 else "0")

            if col == lin:
                t.set_color("black")  # Diagonal (aciertos)
                t.set_fontweight("bold")
            else:
                t.set_color("red")    # Fuera de la diagonal (confusiones)

    for item in text_del:
        item.remove()
    for item in text_add:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    ax.set_title(title, fontweight="bold", fontsize=fz + 4, pad=15)
    ax.set_xlabel("Clase Real (Ground Truth)", fontweight="bold", fontsize=fz + 2, labelpad=10)
    ax.set_ylabel("Clase Predicha", fontweight="bold", fontsize=fz + 2, labelpad=10)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Matriz guardada en: %s y %s", out_path, csv_path)
    return out_path


def get_metrics_table(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Devuelve un DataFrame con las métricas por clase en formato agronómico."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    else:
        labels = sorted(labels)

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    df_metrics = pd.DataFrame(report_dict).T

    df_metrics = df_metrics.rename(columns={
        "precision": "Precisión del Usuario (Precision)",
        "recall": "Precisión del Productor (Recall)",
        "f1-score": "F1-Score",
        "support": "Muestras Reales (Support)",
    })
    return df_metrics


def compute_confusion_matrix_data(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Calcula la estructura de datos completa de la matriz de confusión estilo MATLAB
    para ser consumida directamente por la UI web (pure CSS Grid / Canvas) o APIs.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    else:
        labels = sorted(labels)

    n = len(labels)
    # sklearn confusion matrix: rows = true, cols = pred
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Transpose so rows = predicted, cols = true (MATLAB style)
    cm_trans = cm.T.tolist()

    sum_col = [int(np.sum([cm_trans[r][c] for r in range(n)])) for c in range(n)]
    sum_lin = [int(np.sum(cm_trans[r])) for r in range(n)]
    tot_all = int(np.sum(sum_col))

    # User Accuracy (Precision per row)
    user_acc = []
    user_err = []
    for r in range(n):
        pred_tot = sum_lin[r]
        hit = cm_trans[r][r]
        if pred_tot > 0:
            ok = round((hit / pred_tot) * 100.0, 1)
            err = round(100.0 - ok, 1)
        else:
            ok = 0.0
            err = 0.0
        user_acc.append(ok)
        user_err.append(err)

    # Producer Accuracy (Recall per col)
    producer_acc = []
    producer_err = []
    for c in range(n):
        real_tot = sum_col[c]
        hit = cm_trans[c][c]
        if real_tot > 0:
            ok = round((hit / real_tot) * 100.0, 1)
            err = round(100.0 - ok, 1)
        else:
            ok = 0.0
            err = 0.0
        producer_acc.append(ok)
        producer_err.append(err)

    # Overall Accuracy
    diag_hits = sum(cm_trans[i][i] for i in range(n))
    overall_acc = round((diag_hits / tot_all) * 100.0, 1) if tot_all > 0 else 0.0
    overall_err = round(100.0 - overall_acc, 1) if tot_all > 0 else 0.0

    # Build classification report rows
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    class_metrics = []
    for lbl in labels:
        item = report_dict.get(lbl, {})
        class_metrics.append({
            "class_name": lbl,
            "precision": round(float(item.get("precision", 0.0)) * 100.0, 1),
            "recall": round(float(item.get("recall", 0.0)) * 100.0, 1),
            "f1_score": round(float(item.get("f1-score", 0.0)) * 100.0, 1),
            "support": int(item.get("support", 0)),
        })

    return {
        "labels": labels,
        "matrix": cm_trans,  # [predicted_idx][true_idx]
        "sum_lin": sum_lin,   # row totals (predictions)
        "sum_col": sum_col,   # col totals (real samples)
        "tot_all": tot_all,
        "user_accuracy": user_acc,
        "user_error": user_err,
        "producer_accuracy": producer_acc,
        "producer_error": producer_err,
        "overall_accuracy": overall_acc,
        "overall_error": overall_err,
        "class_metrics": class_metrics,
    }

