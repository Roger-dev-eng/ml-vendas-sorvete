# Previsão de Vendas de Sorvete usando Machine Learning

Este projeto aplica técnicas de Machine Learning (regressão) para prever o número de sorvetes vendidos a partir da temperatura.
Ele demonstra as etapas básicas de um pipeline de ML, do preparo dos dados até a predição via Streamlit.

---
## Conteúdo
- [Importante](#importante)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como funciona](#como-funciona)
- [Azure ML x Scikit-Learn](#azure-ml-x-scikit-learn)
- [Tecnologias utilizadas](#tecnologias-utilizadas)
- [Como rodar o projeto](#como-rodar-o-projeto)
- [Conclusão e aprendizados](#conclusão-e-aprendizados)

## Importante
Este projeto foi inicialmente desenvolvido no Azure Machine Learning (Azure ML) para fins de estudo. 
Depois, o pipeline foi reconstruído localmente usando Python e Scikit-Learn.

---

## Estrutura do Projeto
```
├── app/
│ └── app.py             # Interface Streamlit
│
├── data/
│ ├── raw/               # Dataset original
│ └── processed/         # Dados tratados (gerados)
│
├── models/
│ └── melhor_modelo.pkl  # Modelo treinado (gerado)
│
├── outputs/
│ └── metrics.json       # Métricas (gerado)
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

## Como funciona
O dataset possui duas variáveis principais:
- Temperatura (°C)
- Sorvetes Vendidos

Fluxo completo da pipeline:

1. Pré-processamento
Limpa dados, padroniza colunas e separa train/test (80/20). Não há escalonamento nesta etapa.

2. Treinamento
Testa Regressão Linear, Random Forest e (opcionalmente) XGBoost. O melhor modelo é escolhido pelo menor RMSE.
A Regressão Linear usa `StandardScaler` via `Pipeline` do sklearn, as árvores não.

3. Avaliação
O modelo é avaliado por MAE, RMSE e R², com gráfico True vs Predicted.
A análise SHAP é opcional e desativada por padrão para evitar custo alto.

4. Predição
Suporta predição única, lote via CSV e interface Streamlit.

---

## Azure ML x Scikit-Learn
Este projeto foi inicialmente desenvolvido no Azure Machine Learning, utilizando AutoML para regressão.
Depois, o pipeline foi reconstruído localmente usando Scikit-Learn.

### Resultados no Azure ML (AutoML)
| Métrica                                       | Valor       |
| --------------------------------------------- | ----------- |
| **MAE**                                       | **0.34277** |
| **R²**                                        | **0.99483** |
| **RMSE**                                      | **0.72138** |

### Resultados Localmente no Scikit-Learn
Os valores variam conforme o dataset e o `random_state`. Exemplo de execução local:
| Métrica  | Valor |
| -------- | ----- |
| **MAE**  | 0.5557 |
| **R²**   | 0.9892 |
| **RMSE** | 1.6667 |

---

## Tecnologias Utilizadas
### Cloud
- Azure Machine Learning Studio

### Local
- Python
- Scikit-Learn
- Pandas / NumPy
- Matplotlib / Seaborn
- Streamlit
- Joblib

---

## Como Rodar o Projeto
### Preparação dos dados
Após carregar os dados em `data/raw`, rode:
```bash
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

## Conclusão e Aprendizados
Este projeto permite aprender na prática:
- Construção de um pipeline simples de ML
- Comparação de modelos
- Deploy funcional com Streamlit
- Leitura de métricas (R², MAE, RMSE)
- Visualização de resultados com gráficos
