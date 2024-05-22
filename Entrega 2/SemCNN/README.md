Para explorar a classificação de imagens, indo além do uso de CNNs, foi utilizado o sklearn (RandomForestClassifier) para testar a acurácia de um modelo não baseado em CNN (Convolutional Neural Network), dessa forma foi feito o processamento das imagens de forma a extrair dados referentes a distribuição de valores.

Para fazer o processamento de imagens foi gerado o código do arquivo [process_image.py](./process_image.py) e para ler as imagens do dataset e gerar os dados foi utilizado o arquivo [main.py](./main.py)

## Resultados modelo com RandomForestClassifier

- Tempo de processamento do modelo : < 1 minuto
- Acurácia dos dados : 0.98222 (98.22%)


## Pontos onde o modelo pode ser aprimorado : 

1. Não foi passado nem um hiperparâmetro na configuração do RandomForestClassifier;
2. O último dos parâmetros gerado pela função de processamento de imagens não foi normalizado;