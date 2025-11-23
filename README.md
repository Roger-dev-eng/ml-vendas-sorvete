#  Previsão de Vendas de Sorvete usando Machine Learning

Este projeto aplica técnicas de Machine Learning (Regressão) para prever o número de sorvetes vendidos a partir da temperatura.
Ele demonstra todas as etapas de um pipeline de ML, desde o tratamento dos dados até o deploy de um modelo funcional.

---

## Importante:
Este projeto foi inicialmente desenvolvido no Azure Machine Learning (Azure ML) para fins de estudo, aproveitando:

- Criação e versionamento de datasets

- Experimento automatizado de regressão

- Comparação de modelos com AutoML

- Métricas integradas e registro de artefatos

Após validar a ideia no Azure, o projeto foi reconstruído localmente usando Python, usando:

- Scikit-Learn

- Pandas

- Streamlit
---

##  Estrutura do Projeto
```  
├── app/
│ └── app.py             # Interface Streamlit
│ 
├── data/ 
│ ├── raw/               # Dataset original
│ └── processed/         # Dados tratados
│ 
├── models/ 
│ └── melhor_modelo.pkl  # Modelo treinado
│
├── notebooks/
│ └── ice_cream_ml.ipynb # Notebook demonstrativo
│
├── src/
│  ├── data_prep.py
│  ├── train.py
│  ├── evaluate.py
│  └── predict.py
│
├── README.md
```

---

##  Como funciona

O dataset possui duas variáveis principais:

- Temperatura (°C)

- Sorvetes Vendidos

O fluxo completo de todas as etapas da pipeline:

1. Pré-processamento\
Envolve limpar e padronizar os dados, corrigir inconsistências, normalizar a temperatura com StandardScaler e dividir o dataset em treino e teste (80/20).

2. Treinamento\
Modelos como Regressão Linear, Random Forest e XGBoost são testados, o melhor é escolhido pelo menor RMSE e salvo como melhor_modelo.pkl.

3. Avaliação\
O modelo é avaliado por MAE, RMSE e R², além de um gráfico True vs Predicted para visualizar sua precisão.

4. Predição\
É possível fazer previsões únicas, previsões em lote via CSV e usar uma interface gráfica em Streamlit.
---

## Azure ML X Scikit-Learn
Este projeto foi inicialmente desenvolvido no Azure Machine Learning, utilizando AutoML para regressão.
Depois, o pipeline foi reconstruído localmente usando Scikit-Learn.
A seguir, uma comparação direta das métricas:
### Resultados no Azure ML (AutoML)
| Métrica                                       | Valor       |
| --------------------------------------------- | ----------- |
| **MAE**                                       | **0.34277** |
| **R²**                                        | **0.99483** |
| **RMSE**                                      | **0.72138** |

- O modelo do Azure alcançou R² de 0.99483, indicando que ele explica quase 100% da variação das vendas.
- Os erros (MAE = 0.34, RMSE = 0.72) são extremamente baixos → padrão de modelo quase perfeito.
### Resultados Localmente no Scikit-Learn
Os valores exatos variam conforme o dataset e o random_state, mas geralmente ficam próximos de:
| Métrica  | Valor típico |
| -------- | ------------ |
| **MAE**  | 2.0 – 4.0    |
| **R²**   | 0.85 – 0.95  |
| **RMSE** | 2.5 – 4.5    |

O modelo local é bom, mas não tão preciso quanto o Azure AutoML.
Isso é esperado, porque:
- O treinamento local usa menos modelos (Linear Regression, Random Forest, XGBoost).
- Não há otimização automática de hiperparâmetros avançada.
- O Azure ML usa AutoML + tuning interno, testando dezenas ou centenas de configurações.

---

##  Tecnologias Utilizadas

### Cloud
- Azure Machine Learning Studio
### Local
- Python
- Scikit-Learn
- XGBoost (opcional)
- Pandas / NumPy
- Matplotlib / Seaborn
- Streamlit
- Joblib

---
## Como Rodar o Projeto

### Preparação dos dados
Após carregar os dados em data/ -> raw/, rode: 
``` bash
python src/data_prep.py
```
### Treinamento do modelo
```bash
python src/train.py
```
### Avaliação
```bash
python src/evaluate.py
```
### Executar o Streamlit
```bash
streamlit run app/app.py
```


---
##  Conclusão e Aprendizados

Este projeto foi uma jornada prática no uso de inteligência artificial onde foi possível aprender como:

- Construção completa de um pipeline de machine learning
- Comparação de modelos
- Deploy funcional
- Utilizar o **Azure Machine Learning Studio** para treinar modelos com AutoML de forma automatizada e eficiente.
- Registrar e carregar modelos com **MLflow**.
- Preparar dados corretamente, garantindo que os tipos e nomes de colunas estejam alinhados com o modelo.
- Interpretar métricas de avaliação como R², MAE, RMSE.
- Visualizar os resultados com gráficos e tabelas, facilitando a comunicação dos insights gerados.