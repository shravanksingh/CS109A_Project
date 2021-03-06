# CS109A_Project
This is the repository for Harvard 109A project

Restaurant Recommendation System using Yelp data

Project statement:
Restaurants is one of the major industry that has close ties with humans’ necessity & offers a variety of experiences and food from specialty cuisines (e.g. Mexican, Italian etc.) to taste, décor & services. Yelp users give rating to the restaurant based on their preferences so one rating for a restaurant that all users see. Yelp has a single star rating which is not enough to address different user preferences. Customized rankings are imperatives to any business and people will benefit more from this service. So, we can build a recommendation system that can identify a user’s preferences and provide customized rankings for each individual user. 

This recommender system focuses on predicting the rating that a user would have given to a certain restaurant, which is used to rank all the restaurants including those that have not been rated by the user. 

Data Source:
	The academic dataset (https://www.yelp.com/dataset/challenge) from yelp was downloaded and untarred.

Description of Yelp data:
business.json: Contains business data including location, attributes, and categories. 
review.json: Contains full review text data including the user_id that wrote the review and the business_id the review is written for and the review stars and usefulness. 
user.json: Contains the user's friend mapping and all the metadata associated with the user. 

Data Preparation:
The following steps are used in preparing data for analysis & prediction:
 	
Large Dataset: As the dataset is huge, we only took 100K samples of the observations from each dataset (businesses, users and reviews) to perform the initial EDA.
Combining Data: The sample data was observed to be clean and we merged the dataset based on unique keys.
Filtering: We filtered it down to the restaurants by selecting businesses based on category ‘Food’; for reviews/users we only included the data which was for the restaurants in the sample.
Visualization: We used bar charts, histograms, scatter plots and distribution plots to explore the data and understand the data patterns for the restaurants’ reviews.
Ratings are integers ranging between 1 and 5. The loss function to compare various methods is measured by the root mean squared error (RMSE).

Exploratory Data Analysis:
	We performed Exploratory Data Analysis on the restaurants & the users based on the reviews.
We explored the distribution of ratings with respect to user, business, categories and most reviewed rated restaurants as shown below graphs. We found that most of the users’ rating are left skewed i.e. more people give 4 and 5 star rating than 1 star rating and also found that more users who rate American/Italian restaurants give 4 and 5 rating but users who reviewed Mexican/Chinese restaurants mostly rate with 3 and 4 star.

Baseline Models:
We used Surprise library for Baseline models. Surprise is a Python scikit for building, and analyzing (collaborative-filtering) recommender systems. Various algorithms are built-in, with a focus on rating prediction. 
BaselineOnly is an algorithm predicting the baseline estimate for given user and item 
	Ym = μ + su + sm
where the unknown parameters su and sm indicate the deviations, or biases, of user u and item m respectively from some intercept parameter.

KNNBaseline is a basic collaborative filtering algorithm taking into account a baseline rating.
Memory Based Collaborative filtering:
We used Collaborative filtering. The two primary areas of collaborative filtering are the neighborhood methods and latent factor models. 

Neighborhood methods are centered on computing the relationships between items or, alternatively, between users. The item oriented approach evaluates a user’s preference for an item based on ratings of “neighboring” items by the same user. A product’s neighbors are other products that tend to get similar ratings when rated by the same user. 

Latent factor models (aka SVD) are an alternative approach that tries to explain the ratings by characterizing both items and users on number of factors inferred from the ratings patterns. Latent factor models are based on matrix factorization which characterizes both items and users by vectors of factors inferred from item rating patterns. High correspondence between item and user factors leads to a recommendation. From the results, we can see that prediction accuracy has improved by considering also implicit feedback, which provides an additional indication of user preferences.


MetaClassifier:

We have used multiple models (neighborhoods & SVD) whose individual predictions are combined to classify new examples. Integration should improve predictive accuracy. Each of the models has a mediocre accuracy rate. We would have to increase the importance of the model with high accuracy, and reduce the importance of the models with lower accuracy. To do this in Python, one may use the predicted values as the predictors in a Logistic Regression model, and the corresponding y as the response. Logistic Regression can take the "importance" of each model into account: the "predictors" or models that do well most of the time will have the more significant coefficients.
References
1. How the Netflix prize was won, http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/
2. Matrix factorization for recommender systems, https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
3. Ensembling for the Netflix Prize, http://web.stanford.edu/~lmackey/papers/netflix_story-nas11-slides.pdf
4. Reviews on methods for netflix prize, http://arxiv.org/abs/1202.1112andhttp://www.grouplens.org/system/files/FnT%20CF%20Recsys%20Survey.pdf
5. Advances in Collaborative Filtering from the Netflix prize, https://datajobs.com/data-science-repo/Collaborative-Filtering-%5BKoren-and-Bell%5D.pdf
