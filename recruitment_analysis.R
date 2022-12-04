
#load libraries 
library(readr) 
library(readxl)  
library(ggplot2)   
install.packages("tidytext")
library(tidytext) 
library(forcats) 
library(dplyr) 
library(wordcloud) 

selection_data = read_xlsx("TBS JE Pre-Selection Application (Responses) (3).xlsx")
View(selection_data)

selection_data[["Final Status"]] 
selection_data[["Final Status"]] = ifelse(selection_data[["Final Status"]] == "S",1,0) 
colnames(selection_data)[which(names(selection_data) == "Final Status")] <- "final_selection" 

#factor(selection_data[["final_selection"]])


#text analysis 
#what do they know about TBS JE 
colnames(selection_data)[which(names(selection_data) == "What do you know about TBS JE ?")] <- "information" 
#tidy and plot information  
#custom stop words 
custom_stop_words <- tribble(
       ~ word , ~ lexicon, 
       "tbs","CUSTOM", 
       "je","CUSTOM", 
       "junior","CUSTOM", 
       "marketing","CUSTOM", 
       "services","CUSTOM",  
       "market","CUSTOM", 
       "consulting","CUSTOM", 
       "research","CUSTOM", 
       "firm","CUSTOM", 
       "students","CUSTOM",  
       "professional","CUSTOM", 
       "companies","CUSTOM", 
       "enterprise","CUSTOM", 
       "entity","CUSTOM", 
       "clients","CUSTOM"
       
)
stop_words2 <- stop_words  %>% bind_rows(custom_stop_words) 

tidy_information <- selection_data %>% unnest_tokens(word,information)%>% anti_join(stop_words2)
word_counts <- tidy_information %>% count(word) %>% arrange(desc(n))

# faceting the word counts according to selected or not   

word_counts <- tidy_information %>% count(word,final_selection) %>% group_by(final_selection) %>%
  top_n(10,n) %>% ungroup() %>%
  mutate(word2 = fct_reorder(word,n))  
#visualization 

ggplot(data = word_counts, 
       aes(x = word2, 
           y = n , fill = "final_selection")) + geom_col() + coord_flip()

#convert categorical data as factor 
selection_data[["final_selection"]] <- factor(selection_data[["final_selection"]],levels = c("not selected","selected"))
#visualizing by selected or not  
ggplot(data = word_counts, 
       aes(x = word2 , 
           y = n , fill = final_selection)) + geom_col(show.legend = FALSE,fill = "darkblue") + 
           facet_wrap(~ final_selection, scales = "free_y") + 
           coord_flip()  + 
           ggtitle("TBS JE Definition in terms of Final Selection")
 
#visualization using word clouds 

wordcloud(
  word_counts$word, 
  word_counts$n, 
  max.words = 30, 
  colors = "blue"
)  
 
#plot the regression line between number of words and being selected 
colnames(selection_data)[which(names(selection_data) == "Quality of Hire")] <- "quality_of_hire" 
colnames(selection_data)[which(names(selection_data) == "Average number of Words")] <- "nb_words" 


#model building 
selection_data2 <- selection_data  %>% filter(!is.na(quality_of_hire))
mod1 <- lm(quality_of_hire ~ nb_words,data = selection_data2) 
summary(mod1)    
mod2 <- glm(final_selection ~ nb_words,data = selection_data,family="binomial") 
summary(mod2)
#data visualization 
ggplot(
  selection_data2, 
  aes(x = nb_words,
      y=quality_of_hire)
) +geom_point()  

#data visualization 
ggplot(
  selection_data, 
  aes(x = nb_words,
      y=final_selection)
) +geom_point()  

#comparison cloud between selected and not selected















