library("topicmodels")
library("ldatuning")
library("tm")
asd = read.csv("Go2R.csv",sep= ';')
asd["data"] <- lapply(asd["data"] , as.character)
avector <- list(asd[, "data"])
corpus<-Corpus(VectorSource(c(avector)))

dtm <- DocumentTermMatrix(corpus)
#, control = list(weighting = weightTfIdf, stopwords = FALSE))

result <- FindTopicsNumber(
       dtm,
       topics = seq(from = 2, to = 15, by = 1),
       metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
       method = "Gibbs",
       control = list(seed = 77),
       mc.cores = 4L,
       verbose = TRUE
  )

FindTopicsNumber_plot(result)



#############################
data("crude")

train <- read.csv("train.csv")




train <- read.csv("train_pruebas.csv")



#CTM ala Bernhard Learns Approach
library("magrittr")     #for %$% operator
library("tidyverse")    #R is better when tidy
library("tidytext")     #for easy text  handling
library("ldatuning")    #to get hints for number of topics
library("topicmodels")  #CTM
library("dplyr")        #to make the world a better place


#Tidy the training text, generate a dtm
train$text  <- as.character(train$text)
train$id    <- as.character(train$id)
train$id    <- paste(train$author, train$id, sep="_")

train.tidy <- train %>%
  unnest_tokens(word, text)

train.words <- train.tidy %>%
  count(id, word, sort=TRUE) %>%
  ungroup()



result <- FindTopicsNumber(
  train.dtm,
  topics = seq(from = 2, to = 15, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)



#Find how many topics to consider
system.time(
  topic.number <- FindTopicsNumber(
    train.dtm,
    topics=seq(from=2, to=9, by=1),
    metrics=c("Griffiths2004", "CaoJuan2009",
              "Arun2010", "Deveaud2014"),
    method="Gibbs",
    control=control_list_gibbs,
    mc.cores=4L,
    verbose=TRUE
  )
)
