# SDK móvel para detecção de coação e fraudes — experimento reproduzível

Repositório de apoio ao Trabalho de Conclusão de Curso do MBA em Engenharia de Software
(USP/Esalq), contendo o código, o conjunto de dados sintético e os artefatos necessários para
reproduzir o experimento do motor híbrido de estimativa de risco.

> **Importante:** este repositório apresenta um protótipo acadêmico de laboratório. Ele não representa
> um produto bancário em produção, não realiza diagnóstico de fraude ou coação e não utiliza dados
> reais de clientes, vítimas ou transações financeiras.

## Objetivo

Avaliar experimentalmente um motor de risco que combina:

- regras determinísticas;
- detecção não supervisionada de anomalias com Isolation Forest;
- classificação supervisionada com Random Forest.

A saída é composta por um `risk_score` de 0 a 100 e flags explicativas.

## Estrutura

```text
.
├── experiment.py
├── requirements.txt
├── README.md
├── CITATION.cff
├── LICENSE
├── DATA_LICENSE.md
├── .gitignore
├── docs/
│   ├── DATA_DICTIONARY.md
│   └── METHODOLOGY.md
└── outputs/
    ├── sdk_events_dataset.csv
    ├── sdk_test_results.csv
    ├── metrics.csv
    ├── thresholds.json
    └── split_summary.json
```

## Reprodução do experimento

Recomenda-se Python 3.11 ou superior.

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências e execute:

```bash
pip install -r requirements.txt
python experiment.py
```

Os artefatos serão gravados no diretório `outputs/`.

## Particionamento

A separação é realizada por `session_id`:

- 60% para treinamento;
- 15% para validação;
- 25% para teste final.

Os limiares são calibrados exclusivamente na validação. O teste final não participa do treinamento
nem da seleção dos limiares.

## Resultado reproduzido

| Abordagem | Acurácia | Precisão | Revocação | F1-score | Falsos positivos | Falsos negativos |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 98,04% | 88,28% | 100,00% | 0,9377 | 49 | 0 |
| Isolation Forest | 87,80% | 58,16% | 61,79% | 0,5992 | 164 | 141 |
| Motor híbrido | 98,16% | 89,88% | 98,64% | 0,9406 | 41 | 5 |

Os valores completos, inclusive PR-AUC, estão em `outputs/metrics.csv`.

## Dados

O conjunto `sdk_events_dataset.csv` é **inteiramente sintético** e foi gerado pelo próprio script
com semente pseudoaleatória 42. A proporção de eventos suspeitos foi definida para fins
experimentais e não deve ser interpretada como estimativa de prevalência real.

Consulte `docs/DATA_DICTIONARY.md` para a descrição dos atributos.

## Privacidade

O experimento não contém nomes, CPF, números de conta, números de telefone, credenciais,
conteúdo de mensagens, áudio bruto, coordenadas geográficas reais ou informações de vítimas reais.

## Limitações

Os resultados refletem um ambiente controlado com dados sintéticos. Portanto, não permitem inferir
o desempenho do método em aplicações bancárias reais. Avaliações com maior diversidade
comportamental, dispositivos reais e condições de execução móvel permanecem como trabalhos futuros.

## Trabalho acadêmico

Título: **SDK móvel para detecção de coação e fraudes em transações financeiras**  
Autor: **Lucas Barros Lemos**  
Curso: **MBA em Engenharia de Software — USP/Esalq**  
Ano: **2026**

## Como citar

Uma forma simples de referência é:

> Lemos, Lucas Barros. 2026. SDK móvel para detecção de coação e fraudes em transações financeiras:
> código e dados sintéticos para reprodução experimental. Repositório GitHub.

Após a publicação, substitua esta descrição pelo endereço permanente do repositório.

## Licenças

O código-fonte é disponibilizado sob licença MIT. Os dados sintéticos e a documentação associada
são disponibilizados sob CC BY 4.0, conforme `DATA_LICENSE.md`.
