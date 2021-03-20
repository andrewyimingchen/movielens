# Data setup
#------------------------------------------
library(tidyverse)
library(data.table)
library(caret)
library(recosystem)
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")
?write.csv
write.csv(movielens,"movielense.csv",  row.names = FALSE)

getwd()

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(ratings, movies, test_index, temp, movielens, removed)
#-----------------------------------------------------------------------------------------

# train and test data 
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_data <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_data <- temp %>% 
  semi_join(train_data, by = "movieId") %>%
  semi_join(train_data, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_data)
train_data <- rbind(train_data, removed)

rm(test_index, temp, removed)

# Data exploration 

str(edx)
head(edx)
summary(edx)

n_distinct(edx$rating)

edx %>% group_by(rating) %>% count()



# count of rating
edx %>% ggplot(aes(x = rating)) + 
  geom_bar(fill="#fc0303", alpha=0.5) +
  scale_x_continuous(breaks = seq(0.5,5, 0.5),
                   labels = seq(0.5,5, 0.5)) +
  labs(title = "Movie ratings", y = "Number of Rating", x = "Rating") +
  theme_minimal()

# distribution of movies' avg rating 
edx %>% count(movieId, rating) %>% mutate(score = rating*n) %>% 
  group_by(movieId) %>% summarise(avg = sum(score)/sum(n)) %>%
  ggplot(aes(x = avg)) + 
  theme_minimal() + 
  geom_histogram(bins = 100, fill = "#fc0303", col = "black", alpha=0.5)

# distribution of Movies
## some movies get rated more than others 
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 100, fill = "#fc0303", col = "black", alpha=0.5) + 
  scale_x_log10() +
  ggtitle("Movies")

# not suprising the most rated movies are all blockbusters 
edx %>% 
  count(movieId, title) %>% top_n(n = 15) %>%
  arrange(desc(n))


# Distribution of User
## some users rated more movies than the others, shows clearly that most 
edx %>% 
  count(userId) %>% 
  ggplot(aes(x = n)) + 
  geom_histogram(bins = 100, fill = "#fc0303", col = "black", alpha=0.5)+
  scale_x_log10() +
  ggtitle("Users")


# distribution of number of rating by year 
library(lubridate)
edx <- edx %>% mutate(year = year(as_datetime(timestamp, origin = "1970-01-01")))

n_distinct(edx$year)

edx %>% group_by(year) %>% count()
str(edx)
class(edx$year)
edx %>% ggplot(aes(x = year)) + 
        geom_bar(fill="#fc0303", alpha=0.5) + 
        scale_x_continuous(breaks=seq(1995,2009,1), labels=seq(1995,2009,1)) +
        labs(title = "Distribution of number of movie ratings by year", y = "Number of Rating", x = "Year") +
        theme_minimal()

# number of users by year 
edx %>% count(year, userId) %>% 
  ggplot(aes(x = year)) + 
  geom_bar(fill = "#fc0303", col = "black", alpha=0.5) +
  scale_x_continuous(breaks=seq(1995,2009,1), labels=seq(1995,2009,1)) + 
  labs(title = "Distribution of number of users by year", y = "Number of users", x = "Year") +
  theme_minimal()
# average number of rating by user per year 
edx_1 <- edx %>% group_by(year) %>% count()
a <- edx %>% left_join(edx_1, by = "year") %>% rename(total = n) %>% 
        count(year, userId, total) %>% group_by(year, total) %>% 
        count() %>% mutate(avgn = total/n)
a %>% ggplot(aes(x = year, y = avgn)) + 
      geom_point(col = "#fc0303", alpha=0.5) + geom_line() +
      geom_text(aes(label = round(avgn)), vjust = -1, col = "#fc0303") +
      scale_x_continuous(breaks=seq(1995,2009,1), labels=seq(1995,2009,1)) +
      labs(title = "Distribution of avg number of rating per users by year", y = "Avg number of rating per users", x = "Year") +
      theme_minimal()
# Loss function 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# First Model: Overall average rating 
mu <- mean(train_data$rating)
rmse <- tibble(Method = "Base Model", 
               RMSE = RMSE(test_data$rating, mu))
# Second model: movie effect 
movie_avgs <- train_data %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- test_data %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu +b_i) %>%
  pull(pred)
rmse <- bind_rows(rmse, 
                  tibble(Method = "Base + b_i", 
                         RMSE = RMSE(test_data$rating, predicted_ratings)))
# Third model: user effect 
user_avgs <- train_data %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_data %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse <- bind_rows(rmse, 
                  tibble(Method = "Base + b_i + b_u", 
                         RMSE = RMSE(test_data$rating, predicted_ratings)))

print.data.frame(rmse, digits = 6)
# Fourth Model: regularization 

## checking the model
train_data %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%  
  slice(1:10) %>% 
  pull(title)
# top 10 best movie 
movie_title <- train_data %>% select(movieId, title) %>% distinct()
movie_avgs %>% left_join(movie_title, by = "movieId") %>%
  arrange(desc(b_i)) %>%
  slice(1:10) %>% 
  pull(title)
# top 10 worst movie 
movie_avgs %>% left_join(movie_title, by = "movieId") %>%
  arrange(b_i) %>%
  slice(1:10) %>% 
  pull(title)

# number of rating best movie 
train_data %>% count(movieId) %>% 
  left_join(movie_avgs, by="movieId") %>%
  left_join(movie_title, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n)

# number of rating, worst movie 
train_data %>% count(movieId) %>% 
  left_join(movie_avgs, by="movieId") %>%
  left_join(movie_title, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(n)
## regularization 
lambdas <- seq(0,10, 0.25)

regular <- sapply(lambdas, function(l){
  mu <- mean(train_data$rating)
  b_i <- train_data %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_data %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_data %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_data$rating))
})

tibble(l = lambdas, rmses = regular) %>%
  ggplot(aes(x = l, y = rmses)) +
  geom_point() +
  theme_minimal()

l <- lambdas[which.min(regular)]

mu <- mean(train_data$rating)
b_i <- train_data %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
b_u <- train_data %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))
predicted_ratings <- 
  test_data %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
rmse <- bind_rows(rmse, 
                  tibble(Method = "Regularization", 
                         RMSE = RMSE(test_data$rating, predicted_ratings)))
print.data.frame(rmse, digits = 6)
##

# Matrix Factorization 

#### matrix example 
m <- cbind(1, c(1,NA,3, NA), c(4,5,NA,3), c(NA,2,3,4))
colnames(m) <- c("movie.1","movie.2", "movie.3", "movie.4")
rownames(m) <- rownames(m, do.NULL = FALSE, prefix = "user.")
m
####

#### recosystem 
library(recosystem)
# 1. Transform your train and test data with data_memory()
# 2. Create a model object with Reco()
# 3. Tuning the parameters with $tune()
# 4. Train model with $train
# 5. Predict with $predict 

# transform train data
train_reco <- with(train_data, data_memory(user_index = userId,
                                           item_index = movieId,
                                           rating = rating))
# transform test data
test_reco <- with(test_data, data_memory(user_index = userId,
                                         item_index = movieId,
                                         rating = rating)) 
# create model object 
r <-  recosystem::Reco()
# tuning parameter 
opts <- r$tune(train_reco, opts = list(dim = c(10, 20, 30), 
                                       lrate = c(0.1, 0.2),
                                       costp_l2 = c(0.01, 0.1), 
                                       costq_l2 = c(0.01, 0.1),
                                       nthread  = 4, niter = 10))
# training model 
r$train(train_reco, opts = c(opts$min, nthread = 4, niter = 20))
# testing model 
y_hat_reco <-  r$predict(test_reco, out_memory())
# RMSE
rmse <- bind_rows(rmse, 
                  tibble(Method = "MF", RMSE = RMSE(test_data$rating, y_hat_reco)))
print.data.frame(rmse, digits = 6)

# Verdict 
## validation set 
### Base 
valid <- tibble(Method = "Base Model", RMSE = RMSE(validation$rating, mu))
### Movie effect 
b_i <- train_data %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()))
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
valid <- bind_rows(valid, tibble(Method = "Base + b_i", RMSE = RMSE(validation$rating, predicted_ratings)))
### Movie + user effect 
b_i <- train_data %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()))
b_u <- train_data %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()))
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
valid <- bind_rows(valid, tibble(Method = "Base + b_i + b_u", RMSE = RMSE(validation$rating, predicted_ratings)))
print.data.frame(valid, digits = 6)
#
lambdas <- seq(0,10, 0.25)

regular <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

tibble(l = lambdas, rmses = regular) %>%
  ggplot(aes(x = l, y = rmses)) +
  geom_point() +
  theme_minimal()

l <- lambdas[which.min(rmses)]

mu <- mean(edx$rating)
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)


valid <- bind_rows(valid, tibble(Method = "Regularization", RMSE = RMSE(validation$rating, predicted_ratings)))
print.data.frame(valid, digits = 6)
# MF 
# transform train data
train_edx <- with(edx, data_memory(user_index = userId,
                                   item_index = movieId,
                                   rating = rating))
# transform test data
test_vali <- with(validation, data_memory(user_index = userId,
                                          item_index = movieId,
                                          rating = rating)) 
# create model object 
r <-  recosystem::Reco()
# tuning parameter 
opts <- r$tune(train_edx, opts = list(dim = c(10, 20, 30), 
                                      lrate = c(0.1, 0.2),
                                      costp_l2 = c(0.01, 0.1), 
                                      costq_l2 = c(0.01, 0.1),
                                      nthread  = 4, niter = 10))
# training model 
r$train(train_edx, opts = c(opts$min, nthread = 4, niter = 20))
# testing model 
y_hat_edx <-  r$predict(test_vali, out_memory())
# RMSE
valid <- bind_rows(valid, 
                   tibble(Method = "MF", RMSE = RMSE(validation$rating, y_hat_edx)))
print.data.frame(valid, digits = 6)











