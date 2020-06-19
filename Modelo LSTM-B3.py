
# Importação de bibliotecas de programas
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

# Leitura do arquivo de dados	
cotacoes = pd.read_csv('G:\Meu Drive\Data\VALE3.csv', index_col = 0) 
cotacoesOHLC = cotacoes.copy()
cotacoesOHLC = cotacoesOHLC.dropna()
cotacoesOHLC = cotacoesOHLC[['Open', 'High', 'Low', 'Close']]

# Função de padronização de escala do conjunto de dados
def normescala(df):
    escalaminmax = sklearn.preprocessing.MinMaxScaler()
    df['Open'] = escalaminmax.fit_transform(df.Open.values.reshape(-1,1))
    df['High'] = escalaminmax.fit_transform(df.High.values.reshape(-1,1))
    df['Low'] = escalaminmax.fit_transform(df.Low.values.reshape(-1,1))
    df['Close'] = escalaminmax.fit_transform(df['Close'].values.reshape(-1,1))
    return df

# Padronização do dataframe
cotacoesOHLC_norm = cotacoesOHLC.copy()
cotacoesOHLC_norm = normescala(cotacoesOHLC_norm)

# Segementação do Conjunto de Dados em Treinamento, Validação e Testes 
valid_pct = 10 
test_pct = 10 
sequenc = 20 

def dadoscarga(acoes, sequenc):
    matrizcotacoes = acoes.as_matrix() 
    dadoscotacoes = [] 
    for cont in range(len(matrizcotacoes) - sequenc): 
        dadoscotacoes.append(matrizcotacoes[cont: cont + sequenc])
    dadoscotacoes = np.array(dadoscotacoes);
    valid_tam = int(np.round(valid_pct/100*dadoscotacoes.shape[0])); 
    test_tam = int(np.round(test_pct/100*dadoscotacoes.shape[0]));
    trein_tam = dadoscotacoes.shape[0] - (valid_tam + test_tam);
    x_trein = dadoscotacoes[:trein_tam,:-1,:]
    y_trein = dadoscotacoes[:trein_tam,-1,:]
    x_valid = dadoscotacoes[trein_tam:trein_tam+valid_tam,:-1,:]
    y_valid = dadoscotacoes[trein_tam:trein_tam+valid_tam,-1,:]
    x_test = dadoscotacoes[trein_tam+valid_tam:,:-1,:]
    y_test = dadoscotacoes[trein_tam+valid_tam:,-1,:]
    return [x_trein, y_trein, x_valid, y_valid, x_test, y_test]

x_trein, y_trein, x_valid, y_valid, x_test, y_test = dadoscarga(cotacoesOHLC_norm, sequenc)

# parametros
passosequenc = sequenc-1 
numentradas = 4 
numneurons = 200 
numsaidas = 4
numcamadas = 2
txaprendizado = 0.001
tamlote = 50
numepocas = 100 
trein_tam = x_trein.shape[0]
test_tam = x_test.shape[0]
tf.compat.v1.reset_default_graph()
X = tf.compat.v1.placeholder(tf.float32, [None, passosequenc, numentradas])
y = tf.compat.v1.placeholder(tf.float32, [None, numsaidas])
indsaida = 0;
matrizrandomiz = np.arange(x_trein.shape[0])
np.random.shuffle(matrizrandomiz)

# Função Lote Seguinte
def proxlote(tamlote):
    global indsaida, x_trein, matrizrandomiz  
    inicia = indsaida
    indsaida += tamlote 
    if indsaida > x_trein.shape[0]:
        np.random.shuffle(matrizrandomiz) 
        inicia = 0 
        indsaida = tamlote   
    term = indsaida
    return x_trein[matrizrandomiz[inicia: term]], y_trein[matrizrandomiz[inicia: term]]

# Configuração do modelo LSTM
camadasrede = [tf.contrib.rnn.BasicLSTMCell(num_units=numneurons, activation=tf.nn.elu) 
              for layer in range(numcamadas)]

neuronmulticam = tf.contrib.rnn.MultiRNNCell(camadasrede)
saidasrnn, states = tf.nn.dynamic_rnn(neuronmulticam, X, dtype=tf.float32)
saidasrnnempilh = tf.reshape(saidasrnn, [-1, numneurons]) 
saidasempilh = tf.layers.dense(saidasrnnempilh, numsaidas)
saidas = tf.reshape(saidasempilh, [-1, passosequenc, numsaidas])
saidas = saidas[:,passosequenc-1,:] 

# Função de Custo
perda = tf.reduce_mean(tf.square(saidas - y))

# Otimizador ADAM
otimizad = tf.compat.v1.train.AdamOptimizer(learning_rate=txaprendizado) 
treinamtoproc = otimizad.minimize(perda)

# Treinamento da Rede
with tf.compat.v1.Session() as sesstrein: 
    sesstrein.run(tf.compat.v1.global_variables_initializer())
    for linhadados in range(int(numepocas*trein_tam/tamlote)):
        lote_x, lote_y = proxlote(tamlote) # fetch the next training batch 
        sesstrein.run(treinamtoproc, feed_dict={X: lote_x, y: lote_y}) 
        if linhadados % int(trein_tam/tamlote) == 0:
            trein_errquadmed = perda.eval(feed_dict={X: x_trein, y: y_trein}) 
            valid_errquadmed = perda.eval(feed_dict={X: x_valid, y: y_valid}) 
            print('Iteracao: %4.0f Epoca: %4.0f EQM: treino %.6f validacao %.6f'%(
                    linhadados, linhadados*tamlote/trein_tam, trein_errquadmed
                    ,valid_errquadmed))
  
# Rersultado Predições
    testpredicao_y = sesstrein.run(saidas, feed_dict={X: x_test})

# Verificando predicao 
testpredicao_y.shape

# Plotando Grafico Alvo x Predicao
comp = pd.DataFrame({'Col1':y_test[:,3],'Col2':testpredicao_y[:,3]})
plt.figure(figsize=(10,5))
plt.plot(comp['Col1'], color='blue', label='Alvo')
plt.plot(comp['Col2'], color='red', label='Predicao')
plt.legend()
plt.xlabel("Dias")
plt.ylabel("Preços Padronizados")
plt.show()

# Plotando Grafico de Acuracia: Diferenca Alvo - Predicao
comp = pd.DataFrame({'Col':testpredicao_y[:,3] - y_test[:,3]})
plt.figure(figsize=(10,5))
plt.plot(comp['Col'], color='black', label='Diferença entre Alvo e Predição')
plt.legend()
plt.xlabel("Dias")
plt.ylabel("Diferença de Preços Padronizados")
plt.show()

# Media, Variancia e Desvio Padrao da Acurácia
print('mean    : %.6f ' %(np.mean(comp))) 
print('variance: %.6f ' %(np.var(comp))) 
print('std dev : %.6f ' %(np.std(comp)))
print('correl  : %.6f ' %(np.corrcoef(y_test[:,3],testpredicao_y[:,3])[0,1]))
