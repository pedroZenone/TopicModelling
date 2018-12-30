#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 15:53:01 2018

@author: pedzenon
"""

from LDA_preproc import LDA_preproc
import pandas as pd
import os
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import numpy as np

######  1.Levanto data input y stop words (debe ser ua lista)

# Seteo directorio de trabajo
fileDir = os.path.dirname(os.path.realpath('__file__'))
fileIn= os.path.join(fileDir, 'input')

# Levanto la data y la limpio un poco
data = pd.read_excel(fileIn + '\\' + "2014927443_prueba+R.xlsx")
data.columns = data.iloc[5,]
data = data.drop(data.index[0:6])
data = data.dropna(subset = ['Full Text'])  # saco los mensajes vacios
data = data[["Date","Url","Author","Page Type","Mentioned Authors","Full Text","Full Name","Country Code","Resource Id"]]
data["Date"] = data["Date"].apply(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2].split()[0])
data = data.loc[~ (((data.Date == '2018-10-21') | (data.Date == '2018-10-20') | (data.Date == '2018-10-22')) & (data["Country Code"] == 'ar')) ]

# levanto stop words
lstopwords = pd.read_csv("palabras basura.csv")['palabras'].tolist()       

######  2. Instancio la clase, inicializo
data_preproc = LDA_preproc(data,lstopwords)        
data_preproc.preprocessing()  # arranco el preprocasamiento!      

# me quedo con los tokens que encontre por el momento y la data cruda
token = data_preproc.get_procTokenTweets()
rawTexts = data_preproc.get_rawTweets()

######  3. Me fijo el historico de la data
#data_preproc.exploratoryPlot()

######  4. Me fijo como es la distribucion del largo de palabras y donde considerar un palabra larga
palabrasLargas = data_preproc.get_potencialLotWords(hist = 0,tresh = 15)  # si hist = 0 no muestra grafico

# cargo la lista de palabras que vi potables para truncar
lLargo = pd.read_excel("long.xlsx")['palabras'].tolist()  # me quedo con la raiz de las palabras. Ej: amigosfutbol  -> amigos futbol
data_preproc.truncateLongWords(lLargo)

new_texts = data_preproc.get_procTextTweets()  # ahora voy a tener nuevos texts
token = data_preproc.get_procTokenTweets()

######  5. Analizo las frecuencias de las palabras. Aca puedo ver si tengo palabras que no estan en lematizer o si tengo stopwords a remover

Counter_df = data_preproc.countVectorizer()

######  6. Una vez identificadas las top words o palabras de frecuencia 1 que no aportan o de alta frecuencia que no digan nada:
new_stopwords = pd.read_excel("StopWords2.xlsx")['palabras'].tolist() 
data_preproc.update_StopWords(new_stopwords )

token = data_preproc.get_procTokenTweets()

######  7. Inspecciono las palabras que me llaman la atencion. Puedo ver el mensaje donde aparecieron.

find = data_preproc.inspeccion(r'\bcomer\b')

# si queres hacer zoomin en una cierta palabra y hacer un topico sobre eso:
#data_1 = data_preproc.get_Data()
#data_1.loc[find.indiceRaw.values.tolist()].to_excel("Panal.xlsx")

#####  8.Si veo algo fuera d elugar lo puedo borrar:

data_preproc.delete_containedWord('guillermoprieto')
data_preproc.delete_containedWord('hamburgues')
data_preproc.delete_containedWord('sinaloa')
data_preproc.delete_containedWord('articulo')
data_preproc.delete_containedWord('gobernador')
data_preproc.delete_containedWord('gobierno')
data_preproc.delete_containedWord('gobernar')
data_preproc.delete_containedWord('edomex')
data_preproc.delete_containedWord('policia')
data_preproc.delete_containedWord('diadelamadre')
data_preproc.delete_containedWord('ley')
data_preproc.delete_containedWord('fiscal')
data_preproc.delete_containedWord('foodporn')
data_preproc.delete_containedWord('radiar')
data_preproc.delete_containedWord('taxista')
data_preproc.delete_containedWord('adn')
data_preproc.delete_containedWord('infomigra')
data_preproc.delete_containedWord('escrache')
data_preproc.delete_containedWord('ud')
data_preproc.delete_containedWord('adolescente')
data_preproc.delete_containedWord('laprida')
#####  9. Borro algun author que todos mencionan
data_preproc.delete_MentionedAuthor('@camiirodrigues_')

#####  10. Genero la base para R. Go to R:
texts = data_preproc.get_procTextTweets()
pd.DataFrame(texts,columns = ["data"]).to_csv("Go2R.csv",sep = ";",encoding = "UTF-8")

######  11. LDA. OJO, REFRESCAR token!!!:

token = data_preproc.get_procTokenTweets()  # siempre refresco la data antes de correr todo el LDA!

# inic corpus 4 LDA
dictionary = corpora.Dictionary(token)
corpus = [dictionary.doc2bow(text) for text in token]
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors

for i in range(15):    
    
    # en la gráfica de topicos, marca como el optimo 6 clusters
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=5,random_state  = 5*i)
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'TopicModelling'+str(i)+'.html')
    print("Visualizacion ",str(i), " generada")
#pyLDAvis.show(lda_display)

# Si te gusta como queda el modelo, guardalo por las dudas ;)
fname = 'Bebes5Topics'
lda.save(fileDir + '\\modelos\\'+fname)
lda = models.LdaModel.load(fileDir + '\\modelos\\'+fname)

#############################################
#num_topics = 6
#finaldf = pd.DataFrame([],columns = ["palabras","prob"])
#for num,l in enumerate(aux):
#    
#    df_aux = pd.DataFrame([],columns = ["palabras","prob"])
#    
#    for topicId in range(num_topics):
#        auxx =  pd.DataFrame(l.show_topic(topicId,topn=5),columns = ["palabras","prob"])
#        auxx["topic"] = topicId
#        df_aux = df_aux.append(auxx,ignore_index = True)
#        
#    
#    df_aux["it"] = num
#    finaldf = finaldf.append(df_aux,ignore_index = True)
###################################################################

#####  12.  Le asigno un topico a cada tweet (debe tener una asociacion fuerte, sino le pongo -1). Luego puedo hacer analisis subtopicos!

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=5,random_state  = 5*14)  # una vez que elegi cual de los 8 clusters voy a querer usar me vuelvo a armar el modelo

def num_topic(x,tresh = 0.5):    
    aux = lda.get_document_topics(x)
    rate = aux[np.argmax(([y[1] for y in aux ]))]
    
    umbral_score= tresh
    if(rate[1] > umbral_score):
        return rate[0]          # si el topico elegido es represetativo, devuelvo el valor
    else:
        return -1               # caso contrario deveulvo -1 para mostrar que no es relevante

# si queres ser poco restrictiva y agregar todas las palabras ->  tresh = 0
topic = [num_topic(x) for x in corpus]  # corpus esta definido arriba

reutilizar = data_preproc.get_Data()
reutilizar["Topic"] = topic
reutilizar["Lema Text"] = data_preproc.get_procTextTweets()
#retulizar = reutilizar.loc[reutilizar.Topic == 1]  # aca le pones el topico que te quieras traer

# Para guardalo:
writer = pd.ExcelWriter(r'Bebes5Topics.xlsx', engine='xlsxwriter',options={'strings_to_urls': False})
reutilizar.to_excel(writer)  
writer.save()
writer.close()

#####  13. Interpreto topicos
topics = lda.print_topics(num_words=8)
for topic in topics:
    print(topic)
