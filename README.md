# Equipe-6

# Desenvolvedores

- [Andreia Dourado](https://github.com/andreiadourado)
- [Augusto Cesar](https://github.com/augustces)
- [Joao Victor](https://github.com/JoaoPROFECIA)
- [Mario Umbelino](https://github.com/marioumbelino)
- [Rodrigo Chaveiro](https://github.com/big-rodrigo)
- [Victor Mendes](https://github.com/dvktr)

# Dataset: [Fingers](https://www.kaggle.com/datasets/koryakinp/fingers)

## Descrição
O objetivo do projeto é construir um modelo capaz de contar os dedos através da classificação de imagens e também distinguir entre a mão esquerda e a mão direita.

## Conteúdo
21.600 imagens dos dedos das mãos esquerda e direita.

- Todas as imagens têm 128 por 128 pixels.
- Conjunto de treinamento: 18.000 imagens
- Conjunto de teste: 3.600 imagens
- As imagens são centralizadas pelo centro de massa.
- Padrão de ruído no fundo.

### Etiquetas
Os rótulos estão nos 2 últimos caracteres de um nome de arquivo:
- L/R indica mão esquerda/direita; 
- 0,1,2,3,4,5 indica o número de dedos.

### Observação
Imagens da mão esquerda foram geradas invertendo imagens da mão direita.
