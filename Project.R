


# Load packages
library(readr)
library(dplyr)
library(tm)
library(wordcloud) 

#load the dataset 

jeopardy <- read.csv("jeopardy.csv")

#view and explore the data 
View(jeopardy) 
glimpse(jeopardy) 
head(jeopardy) 

#corpus of categories 

categories <- jeopardy %>%
   filter(round == "Jeopardy!") %>%
   select(ategories = category) 

categories_source <- VectorSource(categories)
categories_corp <- VCorpus(categories_source)

#cleaning the corpus 

clean_corp <- tm_map(categories_corp, content_transformer(tolower))
clean_corp <- tm_map(clean_corp, removePunctuation)
clean_corp <- tm_map(clean_corp, stripWhitespace)
clean_corp <- tm_map(clean_corp, removeWords, stopwords("en")) 

# Create a TDM from the clean corpus
categories_tdm <- TermDocumentMatrix(clean_corp) 
# Create a matrix from the TDM
categories_m <- as.matrix(categories_tdm) 

# Sum the values in each row and sort them in decreasing order
term_frequency <- sort(rowSums(categories_m), decreasing = TRUE) 

# Barplot of the twelve most frequent words
barplot(term_frequency[1:12], col = "orange", las = 2)

















