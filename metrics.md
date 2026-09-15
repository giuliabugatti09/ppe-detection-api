# Métricas do Modelo

Este documento apresenta o desempenho real do modelo treinado, incluindo
limitações conhecidas identificadas tanto na validação quantitativa quanto
em testes manuais durante o desenvolvimento.

## Configuração do treino

| Item | Valor |
|---|---|
| Arquitetura | YOLOv8n (nano) |
| Método | Transfer learning a partir de pesos pré-treinados no COCO |
| Dataset | [ppe_detection_with_yolov8](https://universe.roboflow.com/school-xhq28/ppe_detection_with_yolov8) (Roboflow Universe, CC BY 4.0) |
| Tamanho do dataset | 1.000 imagens |
| Épocas | 50 |
| Resolução de entrada | 640x640 |
| Ambiente de treino | Google Colab (GPU T4) |

## Resultados finais (após 50 épocas)

| Métrica | Valor aproximado |
|---|---|
| mAP@50 | ~0.80 |
| mAP@50-95 | ~0.49 |
| Precision | ~0.87 |
| Recall | ~0.67 |

> Valores lidos das curvas de convergência (`docs/metrics/results.png`).
> O `mAP@50` obtido neste treino específico ficou acima da referência de
> 70% documentada pelo autor original do dataset, mas isso não implica
> superioridade geral do modelo — apenas reflete uma configuração de
> treino e semente aleatória diferentes.

![Curvas de treino](docs/metrics/results.png)

## Desempenho por classe

A matriz de confusão revela uma disparidade importante: o desempenho varia
bastante entre classes, correlacionado diretamente com a quantidade de
exemplos disponíveis no conjunto de validação.

| Classe | Descrição | Instâncias no conjunto de validação | Recall |
|---|---|---|---|
| WHV | Capacete com visor | 131 | 87% |
| WV | Visor | 31 | 77% |
| WH | Capacete (hard hat) | 96 | 76% |
| W | Capacete simples | 19 | **63%** |

**A classe `W` tem o pior desempenho, e também é a classe com menos exemplos
de treino por uma margem considerável (19 instâncias, contra 96-131 das
demais).** Isso é consistente com uma limitação conhecida de datasets
pequenos: classes sub-representadas tendem a ter desempenho pior.

![Matriz de confusão](docs/metrics/confusion_matrix.png)
![Matriz de confusão normalizada](docs/metrics/confusion_matrix_normalized.png)

## Limitação conhecida: falsos positivos em imagens sem EPI

Durante testes manuais (ver histórico de desenvolvimento), uma imagem sem
nenhum capacete visível foi classificada como `WH` com 55% de confiança —
um falso positivo.

A matriz de confusão confirma que isso não foi um caso isolado, mas um
padrão mensurável: em imagens de fundo sem nenhum objeto real (coluna
"background" da matriz), o modelo ainda gera detecções espúrias, sendo:
- 54% delas classificadas como `WHV`
- 29% classificadas como `WH`

**Mitigação aplicada:** o `CONFIDENCE_THRESHOLD` da aplicação foi ajustado
de `0.5` para `0.6` (ver `app/core/config.py`) especificamente para reduzir
a taxa de falsos positivos nessa faixa de confiança limítrofe.

## Limitações gerais e próximos passos

- **Dataset pequeno** (1.000 imagens): mais dados, especialmente da classe
  `W`, provavelmente melhorariam o recall de forma geral.
- **Modelo nano**: escolhido para velocidade de inferência e menor custo
  computacional; um modelo maior (`yolov8s` ou `yolov8m`) provavelmente
  melhoraria a precisão, ao custo de inferência mais lenta.
- **Poucas épocas** (50): as curvas de validação ainda mostram algum
  ruído ao final do treino, sugerindo que mais épocas ou técnicas de
  regularização adicionais poderiam ajudar a estabilizar ainda mais o
  desempenho.
- Este modelo é adequado para fins de demonstração e portfólio, mas não
  foi validado para uso em ambiente de produção real de segurança do
  trabalho, onde o custo de falsos negativos é crítico.