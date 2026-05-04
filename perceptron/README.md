# 🧠 PERCEPTRÓN MULTICONFIGURACIÓN — BREAST CANCER WISCONSIN

```
██████████████████████████████████████████████████████████████████████
█      PERCEPTRÓN MULTICONFIGURACION — BREAST CANCER WISCONSIN       █
██████████████████████████████████████████████████████████████████████

[1/4] Cargando dataset Breast Cancer Wisconsin...
  Muestras:        569
  Características: 30
  Clases: {np.str_('malignant'): np.int64(212), np.str_('benign'): np.int64(357)}
  Train: 455 muestras | Test: 114 muestras

[2/4] Definiendo modelos...

[3/4] Entrenando y evaluando modelos...

  Entrenando: Escalón+Delta Clásico ...

══════════════════════════════════════════════════════════════════════
═                       Escalón+Delta Clásico                        ═
══════════════════════════════════════════════════════════════════════

  Métrica                     Train         Test
  --------------------------------------------
  Accuracy                   0.9824       0.9298
  Precision                  0.9860       0.9571
  Recall                     0.9860       0.9306
  F1                         0.9860       0.9437
  Mcc                        0.9624       0.8513
  AUC_ROC                    0.9988       0.9805

  Tiempo entrenamiento          : 537.90 ms
  Épocas ejecutadas             : 200
  Convergencia en época         : No convergió

  Estadísticas de pesos finales:
    Media=-0.0339  Std=0.1827  Min=-0.3862  Max=0.4846  Norm=1.0178

  Reporte de clasificación (Test):
                  precision    recall  f1-score   support
    
         Maligno       0.89      0.93      0.91        42
         Benigno       0.96      0.93      0.94        72
    
        accuracy                           0.93       114
       macro avg       0.92      0.93      0.93       114
    weighted avg       0.93      0.93      0.93       114

  Entrenando: Escalón+Delta Estocástico ...

══════════════════════════════════════════════════════════════════════
═                     Escalón+Delta Estocástico                      ═
══════════════════════════════════════════════════════════════════════

  Métrica                     Train         Test
  --------------------------------------------
  Accuracy                   0.9978       0.9561
  Precision                  0.9965       0.9718
  Recall                     1.0000       0.9583
  F1                         0.9982       0.9650
  Mcc                        0.9953       0.9064
  AUC_ROC                    0.9993       0.9868

  Tiempo entrenamiento          : 547.69 ms
  Épocas ejecutadas             : 200
  Convergencia en época         : No convergió

  Estadísticas de pesos finales:
    Media=-0.0370  Std=0.1760  Min=-0.3480  Max=0.4889  Norm=0.9850

  Reporte de clasificación (Test):
                  precision    recall  f1-score   support
    
         Maligno       0.93      0.95      0.94        42
         Benigno       0.97      0.96      0.97        72
    
        accuracy                           0.96       114
       macro avg       0.95      0.96      0.95       114
    weighted avg       0.96      0.96      0.96       114

  Entrenando: Sigmoide+Gradient Clásico ...

══════════════════════════════════════════════════════════════════════
═                     Sigmoide+Gradient Clásico                      ═
══════════════════════════════════════════════════════════════════════

  Métrica                     Train         Test
  --------------------------------------------
  Accuracy                   0.9868       0.9737
  Precision                  0.9827       0.9859
  Recall                     0.9965       0.9722
  F1                         0.9895       0.9790
  Mcc                        0.9719       0.9439
  AUC_ROC                    0.9964       0.9957

  Tiempo entrenamiento          : 99.30 ms
  Épocas ejecutadas             : 300
  Convergencia en época         : No convergió

  Estadísticas de pesos finales:
    Media=-0.2859  Std=0.2902  Min=-0.6630  Max=0.3037  Norm=2.2314

  Reporte de clasificación (Test):
                  precision    recall  f1-score   support
    
         Maligno       0.95      0.98      0.96        42
         Benigno       0.99      0.97      0.98        72
    
        accuracy                           0.97       114
       macro avg       0.97      0.97      0.97       114
    weighted avg       0.97      0.97      0.97       114

  Entrenando: Sigmoide+Gradient Estocástico ...

══════════════════════════════════════════════════════════════════════
═                   Sigmoide+Gradient Estocástico                    ═
══════════════════════════════════════════════════════════════════════

  Métrica                     Train         Test
  --------------------------------------------
  Accuracy                   0.9890       0.9737
  Precision                  0.9895       0.9726
  Recall                     0.9930       0.9861
  F1                         0.9912       0.9793
  Mcc                        0.9765       0.9433
  AUC_ROC                    0.9991       0.9924

  Tiempo entrenamiento          : 1053.75 ms
  Épocas ejecutadas             : 300
  Convergencia en época         : 71

  Estadísticas de pesos finales:
    Media=-0.7652  Std=2.0469  Min=-3.8167  Max=5.7224  Norm=11.9694

  Reporte de clasificación (Test):
                  precision    recall  f1-score   support
    
         Maligno       0.98      0.95      0.96        42
         Benigno       0.97      0.99      0.98        72
    
        accuracy                           0.97       114
       macro avg       0.97      0.97      0.97       114
    weighted avg       0.97      0.97      0.97       114

  Entrenando: Sigmoide+PSO Clásico ...

══════════════════════════════════════════════════════════════════════
═                        Sigmoide+PSO Clásico                        ═
══════════════════════════════════════════════════════════════════════

  Métrica                     Train         Test
  --------------------------------------------
  Accuracy                   0.9890       0.9649
  Precision                  0.9930       0.9722
  Recall                     0.9895       0.9722
  F1                         0.9912       0.9722
  Mcc                        0.9766       0.9246
  AUC_ROC                    0.9990       0.9931

  Tiempo entrenamiento          : 350.96 ms
  Épocas ejecutadas             : 200
  Convergencia en época         : 9

  Estadísticas de pesos finales:
    Media=-0.7987  Std=2.8132  Min=-9.0077  Max=7.3850  Norm=16.0174

  Métricas PSO (enjambre):
    Partículas            : 40
    Mejor fitness global  : 0.032153
    Peor pbest            : 0.037855
    Media pbest           : 0.033207 ± 0.001188

  Reporte de clasificación (Test):
                  precision    recall  f1-score   support
    
         Maligno       0.95      0.95      0.95        42
         Benigno       0.97      0.97      0.97        72
    
        accuracy                           0.96       114
       macro avg       0.96      0.96      0.96       114
    weighted avg       0.96      0.96      0.96       114

  Entrenando: Sigmoide+PSO Estocástico ...

══════════════════════════════════════════════════════════════════════
═                      Sigmoide+PSO Estocástico                      ═
══════════════════════════════════════════════════════════════════════

  Métrica                     Train         Test
  --------------------------------------------
  Accuracy                   0.9560       0.9474
  Precision                  0.9649       0.9714
  Recall                     0.9649       0.9444
  F1                         0.9649       0.9577
  Mcc                        0.9061       0.8886
  AUC_ROC                    0.9906       0.9901

  Tiempo entrenamiento          : 228.44 ms
  Épocas ejecutadas             : 200
  Convergencia en época         : 6

  Estadísticas de pesos finales:
    Media=-0.4199  Std=0.8117  Min=-3.9098  Max=0.4147  Norm=5.0054

  Métricas PSO (enjambre):
    Partículas            : 30
    Mejor fitness global  : 0.010740
    Peor pbest            : 0.033292
    Media pbest           : 0.018401 ± 0.004444

  Reporte de clasificación (Test):
                  precision    recall  f1-score   support
    
         Maligno       0.91      0.95      0.93        42
         Benigno       0.97      0.94      0.96        72
    
        accuracy                           0.95       114
       macro avg       0.94      0.95      0.94       114
    weighted avg       0.95      0.95      0.95       114

════════════════════════════════════════════════════════════════════════════════
  TABLA RESUMEN COMPARATIVA
════════════════════════════════════════════════════════════════════════════════
                       Modelo Acc Train Acc Test F1 Test AUC Test MCC Test Tiempo(ms) Convergencia PSO Partículas PSO Mejor Fit
        Escalón+Delta Clásico    0.9824   0.9298  0.9437   0.9805   0.8513      537.9            —            NaN           NaN
    Escalón+Delta Estocástico    0.9978   0.9561  0.9650   0.9868   0.9064      547.7            —            NaN           NaN
    Sigmoide+Gradient Clásico    0.9868   0.9737  0.9790   0.9957   0.9439       99.3            —            NaN           NaN
Sigmoide+Gradient Estocástico    0.9890   0.9737  0.9793   0.9924   0.9433     1053.7           71            NaN           NaN
         Sigmoide+PSO Clásico    0.9890   0.9649  0.9722   0.9931   0.9246      351.0            9             40       0.03215
     Sigmoide+PSO Estocástico    0.9560   0.9474  0.9577   0.9901   0.8886      228.4            6             30       0.01074
════════════════════════════════════════════════════════════════════════════════
  → Guardado: results_summary.csv

[4/4] Generando visualizaciones...
  → Guardado: learning_curves.png
  → Guardado: confusion_matrices.png
  → Guardado: roc_curves.png
  → Guardado: metrics_comparison.png
  → Guardado: training_times.png
  → Guardado: pso_swarm_analysis.png
  → Guardado: weight_distributions.png

══════════════════════════════════════════════════════════════════════
═                        EJECUCIÓN COMPLETADA                        ═
══════════════════════════════════════════════════════════════════════

  Archivos generados:
    📄 learning_curves.png
    📄 confusion_matrices.png
    📄 roc_curves.png
    📄 metrics_comparison.png
    📄 training_times.png
    📄 pso_swarm_analysis.png
    📄 weight_distributions.png
    📄 results_summary.csv
```
