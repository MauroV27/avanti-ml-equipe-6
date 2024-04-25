# Count Finger Accurancy[100%]
[Código Utilizado](https://www.kaggle.com/code/muki2003/count-finger-accurancy-100/notebook)

## Bibliotecas Utilizadas
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
import os
import random
import matplotlib.pyplot as plt
%matplotlib inline
```

***

### Numpy
```python
import numpy as pd
```
> É uma interface dentro do Python que tenta transformar tudo relacionado a listas e matrizes um pouco mais matemático 

Ela fornece suporte para arrays multidimensionais, juntamente com uma ampla variedade de funções matemáticas para operar nesses arrays. NumPy é amplamente utilizado em computação científica e análise de dados devido à sua eficiência e facilidade de uso.

### Pandas
``` Python
import pandas as pd
```
Pandas fornece estruturas de dados e ferramentas de análise de dados de alta performace para Python. Oferencendo estruturas de dados, como o DataFrame, permite a fácil manipulação e análise de conjunto de dados.

### TensorFlow
```python
import tensorflow as tf
```
Uma biblioteca de Código aberto desenvolvida pela Google para machine learning e AI. Oferece estruturas para criar e treinar modelos de machine learning, incluindo deep learning. O TensorFlow é altamente flexível e escalável, e é amplamente utilizado tanto em pesquisa quanto em produção.

### Keras
```python
from tensorflow import keras
```
Keras é uma API de alto nível para construção e treinamento de modelos de deep learning, projetada para ser fácil de usar, modular e extensível. Ela permite que os usuários criem rapidamente protótipos de modelos de deep learning, facilitando o processo de pesquisa e desenolvimento em aprendizado de máquina. Keras pode ser executado em cima de diversas bibliotecas de backend, incluindo o TensorFlow.

### Dense e Activation
```python
from tensorflow.keras.layers import Dense, Activation
```
- Dense é como uma camada de neurônios em uma rede neural. A camada Dense em Keras é completamente conctada, onde cada neurônio na camada está conectado a todos os neurônios da camada anterior. É uma das camadas mais usadas em redes neurais.
- Activation é como um interruptor que controla a saída de cada neurônio em uma rede neural. Ela decide se um neurônio deve ser ativado ou não com base em certas regras. Em reconhecimento de imagens, por exemplo, a função de ativação ode decidir se um neurônio deve "acender" se detectar uma borda em uma imagem. Essas funções ajudam a introduzir não-linearidades nas redes neurais, tornando-as capazes de aprender padrões mais complexos nos dados.

### Adam
```python
from tensorflow.keras.optimizers import Adam
```
Adam é um otimizador que ajuda a ajustar os parâmetros de um modelo de machine learning enquanto ele está sendo treinado. Ele funciona ajustando a taxa de aprendizado de forma adaptativa com base nos gradientes das atualizações dos parâmetros. Em outras palavras, o Adam é bom em ncontrar o caminho crto para minimizar a função de perda do modelo.

### Categorical_Crossentropy
```python
from tensorflow.keras.metrics import categorical_crossentropy
```
É uma função de perda (ou medida) usada para calcular a diferença entre as distribuições de probabilidade previstas pelo modelo e as distribuições reais dos dados. Se as previsões do modelo estiverem próximas da realidade, a perda será baixa; se estiverem longe, a perda será alta.

### ImageDataGenerator
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
É uma ferramenta do Keras para pré-processamento de imagens durante o treinamento de modelos de deep learning. Ela cria lotes de imagens com diferentes transformações, como rotação, ampliação, translação, e assim por diante, para aumentar a quantidade e diversidade dos dados de treinamento. Ajuda o modelo a generalizar melhor e ter um desempenho mais robusto.

### Image
```python
from tensorflow.keras.preprocessing import image
```
Um módulo do Keras que fornece funções para carregar e pré-processar imagens.

### Model
```python
from tensorflow.keras.models import Model
```
Uma classe em Keras usada para instanciar modelos de deep learning. Permite que seja definido a arquitetura do modelo especificando suas camadas e conexões, e então compilar o modelo para treinamento e avaliação.

### Imagenet_utils
```python
from tensorflow.keras.applications import imagenet_utils
```
Fornece ferramentas que ajudam a carregar modelos pré-treinados e a pré-processar imagens de acordo com as necessidades específicas desses modelos. É útil quando se deseja utilizar modelos que já foram trinados em grandes conjuntos de dados.

### OS
```python
import os
```
Fornece funções relacionadas ao sistema operacional, como manipulação de arquivos e diretórios.

### Random
```python
import random
```
Fornece funções para geração de números aleatórios e amostragem de sequências.

### Matplotlib
```python
import matplotlib.pyplot as plt
```
Matplolib é uma biblioteca de visualização de dados em Python. O módulo pyplot fornece uma interface para criar gráficos e visualizações de forma rápida e fácil. O comando `%matplotlib inline` é uma instrução específica do Jupyter Notebook, que permite a exibição de gráficos dentro do notebook.

## Desenvolvendo o Modelo
```python
trainpath = os.listdir("../input/fingers/train")
testpath = os.listdir("../input/fingers/test")
```
Utilizando a biblioteca `os`, as duas linhas estão listando todos os arquivos e diretórios dentro `"../input/fingers/train"`, armazenando-os nas variáveis _trainpath_ e _testpath_.

### Criando Lista com Nome das Imagens
```python
traindata = ['../input/fingers/train/' + i for i in trainpath]
testdata = ["../input/fingers/test/" + i for i in testpath]
```
Para facilitar o acesso e manipulação dos arquivos, é criado listas de caminhos completos para todos os arquivos dentro dos diretórios de treinamento e teste, através do conceito *list comprehension*. Ao fazer isso, podemos facilmente carregar os dados de treinamento e teste em um formato adequado para o treinamento e validação do modelo.

### Convertendo Lista para DataFrames
```python
traindata = pd.DataFrame(traindata, columns=['Filepath'])
testdata = pd.DataFrame(testdata, columns=['Filepath'])
```
Neste código, a lista é transformada em DataFrames do pandas, onde cada imagem é colocado em uma linha. Cada coluna tem apenas uma coluna "Filepath".

### Separando Valor Alvo do Nome do Arquivo
```python
traindata['target'] = traindata['Filepath'].apply(lambda a: a[-6:-5])
testdata['target'] = testdata['Filepath'].apply(lambda a: a[-6:-5])
```
No nome do arquivo, há um número na qual indica o valor exibido pela mão. Nesse código, é separado esse número do nome do arquivo, realizando a classificação das imagens para processamento posterior.

### Imagem
```python
from IPython.display import Image
Image(filename=traindata.Filepath[0], width=300,height=300) 
```
Exibe a primeira imagem do dataset. 

### Pré-processamento de Dados Usando MobileNet
```python
ds_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,validation_split=0.1)
```
É usado `ImageDataGenerator` para gerar lotes de dados de imagem para treinamento de modelos de rede neural convolucional (CNN), usando arquitetura MobileNet. A função `preprocess_input`, da arquitetura MobileNet, do TensorFlow, realiza o pré-processamento necessário para as imagens antes de alimentá-las para a rede neural MobileNet. Esse pré-processamento geralmente envolve normalização e redimencionsamento das imagens para que estejam em um formato adequado para o modelo. O argumento `validation_split=0.1` define a fração dos dados de treinamento que serão reservados para validação durante o treinamento. Neste caso, 10% dos dados serão usados para validação e o restante para treinamento.

### Conjunto de Dados para Treinamento, Validação e Teste.
```python
train_ds = ds_generator.flow_from_dataframe(dataframe=traindata,x_col='Filepath',y_col='target',target_size=(224, 224),color_mode='rgb',class_mode='categorical',batch_size=16,subset='training')
val_ds = ds_generator.flow_from_dataframe(dataframe=traindata,x_col='Filepath',y_col='target',target_size=(224, 224),color_mode='rgb',class_mode='categorical',batch_size=16,subset='validation')
test_ds = ds_generator.flow_from_dataframe(dataframe=testdata,x_col='Filepath',y_col='target',target_size=(224, 224),color_mode='rgb',class_mode='categorical',batch_size=16)
```
Utilizando `flow_from_dataframe`, é configurado geradores de fluxo de dados para os conjuntos criados anteriormente.
- `target_size` - define o tamanho das imagens de entrada. No caso, o dataset contém apenas imagens 224x224 pixels. 
- `color_mode` - define o mode de cor das imagens. No caso, "rgb".
- `class_mode` - define o módulo de classificação dos rótulos. 'categorical' indica que os rótulos são codificados omo verotes *one-hot*. Ou seja, se houver N classes, cada rótulo será um vetor com N elementos, onde apenas um elemento é 1 e os outros são 0.
- `batch_size` - define o tamanho do lote de dados a ser alimentado no treinamento, validação e teste.
- `subset` - especifica se os dados são para treinamento, validação ou teste.

Esses Geradores são usados para alimentar dados para treinar e avaliar um modelo de rede neural. 

### Modelo e Resumo MobileNet
```python
mobile = tf.keras.applications.mobilenet.MobileNet()

mobile.summary()
```
Mostra informações detalhadas sobre cada camada do modelo, sendo útil para entender a composição e a complexidade do modelo antes de usá-lo para classificação de imagens.

```python
x = mobile.layers[-6].output

output = Dense(units=6, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=output)

model = Model(inputs=mobile.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

- `x = mobile.layers[-6].output` - remove as últimas 5 camadas do modelo MobileNet

- `output = Dense(units=6, activation='softmax')(x)` - adiciona uma nova camada densa. A função de ativação `'softmax'` ajuda a garantir que as saídas da camada estejam na forma de probabilidades.

- `model = Model(inputs=mobile.input, outputs=output)` - cria um novo modelo, usando a classe 'Model'.

- `model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])` - momento em que o modelo recém-criado é compilado.
  
    - `optimizer=Adam(learning_rate=0.001)` - usado para atualizar os pesos do modelo durante o treinamento. O parâmetro `learning_rate=0.001` define a taxa de aprendizado, que controla o tamanho das atualizações feitas nos pesos durante o treinamento. 
  
    - `loss='categorical_crossentropy'` - função de perda que será utilizada para avaliar a diferença entre as previsões do modelo e os rótulos verdadeiros durante o treinamento. A função `'categorical_crossentropy'` é geralmente usada para problemas de classificação multiclasse, onde as saídas do modelo são interpretadas como probabilidades para cada classe. 

    - `metrics=['accuracy']` - estas são as métricas que serão usadas para avaliar o desempenho do modelo durante o treinamento e a avaliação. A métrica `'accuracy'` mede a proporção de exemplos classificados corretamente pelo modelo. 

```python
model.fit(train_ds, validation_data=val_ds, verbose=1, epochs=2)
```
Código executa o treinamento do modelo usando os dados de treinamentro e validação, através da função `fit()`.
- `verbose=1` - controla a quantidade de informações que são exibidas durante o treinamento. O valor de 1 significa que o progresso será exibido em detalhes durante o treinamento.

- `epochs=2` - define o número de vezes que o modelo passará por todo o conjunto de treinamento. Cada época (vezes) consiste em uma passagem por todos os exemplos de treinamento. No caso, o modelo será treinado por duas épocas.

```python
model.summary()
```
Exibe o resumo do modelo treinado. 

### Avaliação do Modelo com Dados de Teste
```python
model.evaluate(test_ds)
```
A função `evaluete` avalia o modelo usando um conjunto de dados, neste caso, o conjunto de dados de testes. Ela calcula a função de perda e quaisqer métricas especificadas durante a compilação do modelo nos dados (nesse caso, *accuracy*).
- output - retorna uma lista de valores, na qual o primeiro elemento é a perda calculada nos dados de teste e os demais elementos, as métricas.

### Salvando o Modelo
```python
model.save('./CountFinger1.h5')
```
O modelo é salvo em um arquivo HDF5, que é frequentemente utilizado para armazenar grandes quantidades de dados científicos de maneira eficiente. O arquivo contém todos os detalhes do modelo, como arquitetura de rede, pesos treinados e as configurações de compilação. 

