# Reddit-API-Clasffication-Model

## Problem Definition

We look to explore classification models within the realm of natureal language processing (NLP).

The problem we are seeking to answer is whether or not we can build a classfication model to accurately predict if a post belongs to one subreddit or the other. Should we be able to, we take a closer look at which model(s) serves this purpose the best and why.

The stakeholders for this problem are the respective subreddit mods. This also extends beyond reddit to other forums and websites. We are looking to help these people to easily fish out posts which users have posted to the wrong subreddit and allow them to direct these users to the alternate subreddit where they can get access to the intended audience.

Success of our model will depend on the accuracy scores from each model which will be done on 'unseen' data via a train-test split of the dataset. We can use accuracy in this case over other metrics such as sensitivity or specificity as there are no better/ worse implications for type 1 or type 2 errors. Assuming we assin the value of 1 for askscience, a false positive would be a post that is classified as belonging to ask science when it actually belongs to AskHistorians. A false negative would be an askscience post that has been classified to belonging to AskHistorians. We can think of these as inverse to each other and of equal severity and as such, accuracy will be a good metric to score our models.

The motivation behind understanding which models work and what parameters work the best is that we can scale the models to fit different scenarios/ datasets if needed.

## Data Gathering

We have scraped data from the 2 subreddits in a seperate notebook with the following fields of data:
* Subreddit
* Name
* Title
* Selftext
* Score

In total, we have approximately **1000 unique posts per subreddit**. We have ensured that the posts were not repeated in the other notebook.

## Data Exploration and Cleaning

We look to clean the data before passing it through our models. The things we will be focused on include:
* Removing hyperlinks
* Removing HTML characters
* Converting words to lowercase
* Removing stopwords, including words that occur frequently in both subreddits
* Lemmatizing the words

## Modelling the Data

We explore the following models, both with countvectorizer and TF-IDF vectorizer and a train-test split:
* Logistic Regression
* Naive Bayes
* K Nearest Neighbors
* Decision Trees
 
 These are the results:
 
 ![image](https://user-images.githubusercontent.com/49399188/123557893-0b6b2600-d7c6-11eb-9192-8871e1228f8f.png)
 
 We select the Logistic Regressoin, Multinomial Naive Bayes and K Nearest Neighbors for further examination (all with TF-IDF Vectorizer)

 This is a summary of our models:
 
 ![image](https://user-images.githubusercontent.com/49399188/123558006-977d4d80-d7c6-11eb-8481-27e324d88d55.png)
 
 We see that Naive Bayes performed the best on test data (lowest varaince) but KNN presented the lowest bias.
 
 We also look at the misclassifications:
 
 ![image](https://user-images.githubusercontent.com/49399188/123558025-c1cf0b00-d7c6-11eb-8f02-c6dd8e091e00.png)
 We can see that both Naive Bayes and KNN more confidently predicted the classes with fatter tails in the chart. However, logistic regression did well in minimising the magnitude of misclassifcation.

Across all models,:
Misclassified science posts contained topics surrounding geography (lakes, rivers, volcano, island etc.)
Misclassified History posts contained technical words that were in a different context (milk formula, doctor, medical, interchangeable currency, popular displacement)

## Best Predictors
![image](https://user-images.githubusercontent.com/49399188/123558072-022e8900-d7c7-11eb-93c5-f5aaf02e9a93.png)

These were the top 10 best predictors for each subreddit for both the logistic regression and naive bayes. 
We can see that the top words for science is more or less the same. The most interesting one is the word 'cause'. This is an unlikely word to appear but after some consideration, this makes sense. Science is described by the [Cambridge Dictionary](https://dictionary.cambridge.org/dictionary/english/science) as the careful study of the structure and behaviour of the physical world, especially by watching, measuring, and doing experiments, and the development of theories to describe the results of these activities. A large part of science includes the understanding of causal activities and thus, this is why the word pops up and is useful in our model. Another interesting thing is that we see 'difference' and 'different' appear as well. This links back to our logic that for science topics, relationships between certain phenomenon is a common question to ask and thus, these words being significant. However, when we look at the top words for history, there is a vast difference. To understand this, we have to look at the difference between the Naive Bayes and Logistic Regression models.

The second column shows the top terms you would expect to see in the AskHistorians subreddit. These are largely surrounding different civilisations and 'empires' - we see roman, american, german. An interesting one is the word 'th'. This could possibly be more another word lemmatized through our text cleaning earlier but after further exploring, it seems that words such as 'the', 'them' etc retain their form. This is most likley the result of us dropping numbers from phrases such as '17th century' which leaves behind 'th'.

There is however, a divergence in the top predictors for History.

Naïve Bayes assumes all the features to be conditionally independent. So, if some of the features are in fact dependent on each other (in case of a large feature space), the prediction might be poor.
Logistic regression splits feature space linearly, and typically works reasonably well even when some of the variables are correlated.

Given this, the output for logistic regression gives us more context and given that some words might be correlated to others, the top words for logistic regression are the ones that are the best predictors with the context of other words whereas for naive bayes, these are the words that give us the best predictors independently.

Source: [Comparison between Naive Bayes and Logistic Regression](https://dataespresso.com/en/2017/10/24/comparison-between-naive-bayes-and-logistic-regression/)

 ## Findings

We have shown that we are indeed able to build a model that can help us with our problem at hand and has an accuracy of ~94% in classifying unseen data. This then begs the question, which model should we use?

### The 'best' model
There is no clear answer for this and this depends on what we are seeking to achieve.

The first step is clear, the TF-IDF vectorizer is the vectorizer of choice across all our models. This vectorizer helps us tease out relevance of words instead of simply just the count of words. If we are unable to clearly pre-process our data, this is even more helpful. Even if we can pre-process our data and clean it to a considerable extent, the TF-IDF vectorizer still serves us well. It even helps us 'scale' the data to fit into models such as K Nearest Neighbors where normal standard scaling will prove to be a challenge. Count vectorizer suffers from the following shortfalls:
* Its inability in identifying more important and less important words for analysis.
* It will just consider words that are abundant in a corpus as the most statistically significant word.
* It also doesn't identify the relationships between words such as linguistic similarity between words.

**It is clear that TF-IDF vectorizer is the pre-processor for our data**

Next, what classification model should we use? 

If we are purely concerned with the highest accuracy then the Naive Bayes model is the go-to model of choice. However, if we are looking for a healthy balance between bias-variance, the K Nearest Neighbors model showed the smallest degree of overfitting as compared to the other 2 models. The logistic regression overfitted by 6.13%, the naive bayes overfitted by 5.60% but the K Nearest Neighbors Classifier overfitted by only 2.6%.

Does this mean that the logistic regression is redundant? Hardly so. The logistic regression performed the worst most possibly due to the fact that we had a relatively small training set. In a [paper written by Professor Andrew Ng and Professor Michael I Jordan](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf), it provided a mathematical proof of error properties of both the logistic regression and naive bayes. They concluded that when the training size reaches infinity, the discriminative model: logistic regression performs better than the generative model (naive bayes). 

Generative classifiers learn a model of the joint probability of the inpputs (x) and the label (y) and make their predictions using Bayes rules to calculate the probability of y given x and then picking the most likely label y. Discriminative classifiers model the posterior probability of y given x directly, or learn a direct map from inputs x to the class labels. One of the compelling reasons for using a logistic regression over naive bayes as articulated by Vapnik is that one should solve the classification problem directly and never a more general problem as an intermediate step (such as modelling probability of x given y).

The generative model (Naive Bayes) reaches the asymptotic solution for fewer training sets than the discriminative model (Logistic Regression). This behavior is best represented by the experiment conducted by Ng & Jordan where they did predictions for 15 datasets from the UCI machine learning repository. In some cases the training sample was not large enough for logistic regression to win.

In mathematical analysis, asymptotic analysis, also known as asymptotics, is a method of describing limiting behavior. In mathematical statistics and probability theory, asymptotics are used in analysis of long-run or large-sample behaviour of random variables and estimators.

In short Naive Bayes has a higher bias but lower variance compared to logistic regression. If the data set follows the bias then Naive Bayes will be a better classifier. Both Naive Bayes and Logistic regression are linear classifiers, Logistic Regression makes a prediction for the probability using a direct functional form where as Naive Bayes figures out how the data was generated given the results.

It is clear now when we should use logistic regression or naive bayes, what about K Nearest Neighbors? As mentionted above, it was the model that showed the lowest degree of overfitting. However, the model was successful likely due to the extent of our pre-processing and how we were able to effectively remove irrelevant words. In a [paper](https://core.ac.uk/download/pdf/82438337.pdf) by Bruno Trstenjak, Sasa Mikac and Dzenana Donko in the 69th edition of Procedia Engineering, they did an in-depth study on KNN with TF-IDF Based Framework for Text Categorization. The framework they presented gave good results and confirmed their initial expectations that the model would be effective 

However, tests have shown that the embedded algorithm is sensitive to the type of dicuments. They classified text documents into Sport, Politics, Finance and Daily News. Sport showed the highest accuracy of 0.92, followed by Politics with 0.90, Finance with 0.78 and Daily News with 0.65. This was due to the fact that the document contents in the category Daily News contained a lot of 'unusable words', words that are often repeated and do not have important weight but have an adverse impact on KNN. The analysis of documents showed that the amount of unusable words in documents has a significant impact on the final quality of classification. because of this, it is important to improve the preprocessing of data for better results.

Given that this is a lot of information to take in, we present a simple flowchart that serves as a guide to think about what the appropriate model to use is.
![flowchart](https://user-images.githubusercontent.com/49399188/123557978-6dc42680-d7c6-11eb-9405-9d2773e2b6d8.JPG)

### Future steps
A potetial improvement for the project would be to examine the models with larger datasets. This can truly test the assumption that as training datasets grow, logistic regressions will perform better.

Another improvement will be to use more similar categories to really put our pre-processing to the test. We have been quite fortunate to deal with technical topics with many 'usable' words. Should we be able to work on more generic categories, we can potentially test to see if K Nearest Neighbors still performs well and to what extent pre-processing can improve our results.
