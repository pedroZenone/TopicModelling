#install.packages("tm")  # for text mining
#install.packages("SnowballC") # for text stemming
#install.packages("wordcloud") # word-cloud generator 
#install.packages("RColorBrewer") # color palettes

# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

data = read.csv("TextsTopics.csv",sep= ';')
data = data.frame(data)

for (x_topic in unique(data["topic"])$topic){

  topic = data[data$topic == x_topic,]["text"]
  topic_p <- lapply(topic , as.character)

  corpus<-Corpus(VectorSource(c(topic_p)))
  
  dtm <- TermDocumentMatrix(corpus)
  m <- as.matrix(dtm)
  v <- sort(rowSums(m),decreasing=TRUE)
  d <- data.frame(word = names(v),freq=v)
  
  png(paste(c("wordcloud_packages",x_topic,".png"), collapse=""), width=600,height=400)
  wordcloud(words = d$word, freq = d$freq, min.freq = 1,
            max.words=200, random.order=FALSE, rot.per=0.35, 
            colors=brewer.pal(8, "Dark2"))
  dev.off()
}
