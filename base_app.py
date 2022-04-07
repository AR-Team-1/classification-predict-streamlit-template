"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("TWEET CLASSIFIER")

	# Creating sidebar with radio buttons -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Data Visualization", "Meet The Team", "About Us"]

	with st.sidebar:
		image = Image.open('resources/imgs/logo_thingy.jpeg')
		st.image(image, caption="", width=150)
		st.markdown("""## Navigation""")
		selection = st.radio("Explore Our Options", options)
	

	# Building out the "Meet The Team" page
	if selection == "Meet The Team":
		st.subheader("Meet the Team")

		image = Image.open('resources/imgs/helloimnick._meet the team.jpg')
		st.image(image, caption="Photo Credit: Hello I'm, unsplash.com")

		# You can read a markdown file from supporting resources folder
		st.markdown("""
		
		Our team consists of 5 talented data scientists and developers from various parts of Africa. These are:
		- Lungisa Joctrey (South Africa)
		- Christian Miri  (Nigeria)
		- Precious Orekha (Nigeria)
		- Ibrahim Isarki  (Nigeria)
		- Caleb Tanko     (Nigeria)

		""")

	# Building out the "About Us" page
	if selection == "About Us":
		st.subheader("PLICC Analytics")
		#Company logo

		image = Image.open('resources/imgs/logo_thingy.jpeg')
		st.image(image, caption='Photo Credit: Ibrahim Sarki')
		# You can read a markdown file from supporting resources folder
		st.markdown("""
		PLICC ANALYTICS specializes in Information Technology Services. We take 
		data and arrange it in such a way that it makes sense for business and individual users. We also build and train models that are capable of solving a wide range of classification problems. 
		
		Our team of leading data scientists work tirelessly to make your life and the life of your customers easy.
		"""
		)

		st.markdown(""" 
		For more info:

		email: info@pliccanalytics.com
		""")

	# Building out the "Data Visualisation" page
	if selection == "Data Visualization":
		st.subheader("Data Visualization")
		#displays an image 
		image = Image.open('resources/imgs/fabio-unsplash,DataViz.jpg')
		st.image(image, caption='Photo Credit: Fabio, unsplash.com')

		#creating visual options for Data Visualisation
		sentiments = ["All", "Positive Sentiment", "Neutral Sentiment", "Negative Sentiment"]


		sentiment = st.radio("Climate Change Sentiments", sentiments)

		#EDA for all sentiments
		if sentiment == "All":
			st.info("You are Viewing: Data Analysis for all climate change sentiment groups")
			st.markdown("""
			When we receive our raw data we have to clean it and then do a thorough investingation on it so that 
			we make sense. We refer to that investigation process as Exploratory Data Analysis (EDA). 
			
			""")

			image = Image.open("resources/imgs/Otherwc.jpeg")
			st.image(image,use_column_width=True)
			st.markdown("""
			The above word cloud, shows us that overall,"climate change", "climate", "trump ", "global warming" and "threaten"
			are words that are often used in the tweets contained in our dataset. It appears that surrounding climate change,
			people have lot of elements they attach to it, in order to bolt their stance on whether they believe in climate change or not.

			

			""")

			#Image of data distribution
			image = Image.open("resources/imgs/distribution_sentiment.png")
			st.image(image, use_column_width=True)

			st.markdown("""

			In the image above we learn that our data is imbalanced and so we had to balance the data
			so that our models can be trained effectively. This means, those who believe in climate change as fact
			outnumber those who no not believe in climate change and those who are indifferent to it. This will cause the image
			to be baised towards the majority. Balancing the data will reduce chances for the model to side the data points it sees the most.
			
			Below is the graph that shows a balanced dataset.
			 
			 """)

			image = Image.open("resources/imgs/balance_data_distribution.png")
			st.image(image, use_column_width=True)

		#EDA for positive sentiments
		if sentiment == "Positive Sentiment":
			st.info("You are Viewing: Data Analysis for believers in climate change")
			st.markdown("""
			Below we take a look at the word cloud for words mentioned by people who believe in climate change.
			And from it we seek to draw insight on this group and on what influences their sentimental position.
			""")
			image = Image.open("resources/imgs/positive wc.jpeg")
			st.image(image)
			st.markdown("""
			In the wordcloud above we can see that, "climate change", "global warming","believe","evidence", "trump", and 
			"climate" have the most mentions in the believing group. It seems like, a number of people perhaps blamed Donald Trump or 
			appealed to Trump to do something in order to improve the situation with regards the climate chang topic.

			""")



		#EDA for negative sentiments
		if sentiment == "Negative Sentiment":
			st.info("You are viewing: Data Analysis for non-believers in climate change")

			st.markdown("""
			Below we take a look at the word cloud for words mentioned by people who do not believe in climate change.
			And from it we seek to draw insight on this group and on what influences their sentimental position.
			""")
			image = Image.open("resources/imgs/negativewc.jpeg")
			st.image(image)
			st.markdown("""
			In the negative sentiment group a lot of people mentioned, "climate change", "global warm","hoax","al gore","manmade", "chinese",
			"snowflakes","science" and "prove."

			Noticing the mention of Al Gore, shows that those who do not believe in climate change are likely to blame Al Gore for inventing the idea.
			most of them believe that climate is a hoax.
			""")
			

		#EDA for neutral sentiments
		if sentiment == "Neutral Sentiment":

			st.markdown("""
			Below we take a look at the word cloud for words mentioned by people who are neutral in the climate change topic.
			And from it we seek to draw insight on this group and on what influences their sentimental position.
			""")
			st.info("You are Viewing: Data Analysis for people who are indifferent about climate change")
			image = Image.open("resources/imgs/neutralwc.jpeg")
			st.image(image)
			st.markdown("""
			In the neutral group a lot of people mentioned, "climate change", "global warming","trump","warm",
			"believe" and "bullshit." While neutral in sentiment, this group still enganged in the topic of climate change
			with passion. 
			""")

		

		
	
		# You can read a markdown file from supporting resources folder
		st.markdown("")

		

	# Building out the "Information" page
	if selection == "Information":
		#st.info("General Information")
		st.subheader("General Information")
		#
		image = Image.open('resources/imgs/ux-indonesia_.jpg')
		st.image(image, caption='Photo Credit: ux, unsplash.com')

		# You can read a markdown file from supporting resources folder
		st.markdown("""
		The end goal of this research is to look at the tweets from individuals and determine if that particular person believes in the climate change or not. We have created and trained several models that can do this task.  

		Below is the data used to train the models.
		
		""")

		st.subheader("Raw Twitter data and label")
		#showing raw data
		st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the prediction page
	if selection == "Prediction":
		#loading an image
		st.subheader("Prediction with ML Models")
		image = Image.open('resources/imgs/kaitlyn-bakerPrediction.jpg')
		st.image(image, caption='Photo Credit: Kaitlyn Baker, unsplash.com')
		

		# Creating a text box for user inputs
		tweet_text = st.text_area("Try it","Type your text here")
		models = ["RBF SVM","Logistic Regression", "Support Vector Classifier(SVC)"]

		#creating a selection for panel fo models
		choice = st.radio("Choose A Model", models)

		#Setting conditions when the user chooses "RBF SVM"
		if choice == "RBF SVM":
			st.info("""
			You chose RBF SVM. 
			
			In machine learning, the Radial Basis Function (RBF) kernel, or RBF kernel, is a popular kernel 
			function used in various kernelized learning algorithms.
			""")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/RBF SVM.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				#makin prediction is humanly understable

				if prediction == [0]:
					pred_class = "Likely not a believer in climate change"
				else:
					pred_class = "Likely a believer in climate change"	


				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(pred_class))



		#Setting conditions when the user chooses "Logistic Regression"
		if choice == "Logistic Regression":
			st.info("""
			You chose Logistic Regression

			Logistic regression is a statistical analysis method to predict 
			a binary outcome, such as yes or no, based on prior observations of a data set. 
		
			""")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				#makin prediction is humanly understable

				if prediction == [0]:
					pred_class = "Likely not a believer in climate change"
				else:
					pred_class = "Likely a believer in climate change"	


				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(pred_class))

		#Setting conditions when the user chooses "SVC"
		if choice == "Support Vector Classifier(SVC)":
			st.info("""
			You chose SVC

			Support Vector Classifier is a nonparametric clustering algorithm that does not make any 
			assumption on the number or shape of the clusters in the data.
			""")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/SVC.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				#makin prediction is humanly understable

				if prediction == [0]:
					pred_class = "Likely not a believer in climate change"
				else:
					pred_class = "Likely a believer in climate change"	


				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(pred_class))



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
