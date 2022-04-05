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
news_vectorizer = open("resources/tfidfvect.pkl","rb")
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

	# Building out the "Data Visualisation" page
	if selection == "Data Visualization":
		st.subheader("Data Visualization")
		#displays an image 
		image = Image.open('resources/imgs/fabio-unsplash,DataViz.jpg')
		st.image(image, caption='Photo Credit: Fabio, unsplash.com')

		#creating visual options for Data Visualisation
		sentiments = ["All", "positive", "negative", "neutral"]

		sentiment = st.radio("Climate Change Sentiments", sentiments)

		#EDA for all sentiments
		if sentiment == "All":
			st.info("You are Viewing: Data Analysis for people")

		#EDA for positive sentiments
		if sentiment == "positive":
			st.info("You are Viewing: Data Analysis for believers in climate change")
		#EDA for negative sentiments
		if sentiment == "negative":
			st.info("You are viewing: Data Analysis for non-believers in climate change")
		#EDA for neutral sentiments
		if sentiment == "neutral":
			st.info("You are Viewing: Data Analysis for people who do give a shit about climate change")


		

		
	
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

		Below is the data used to train the model.
		
		""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the prediction page
	if selection == "Prediction":
		#loading an image
		st.subheader("Prediction with ML Models")
		image = Image.open('resources/imgs/kaitlyn-bakerPrediction.jpg')
		st.image(image, caption='Photo Credit: Kaitlyn Baker, unsplash.com')
		

		# Creating a text box for user input
		tweet_text = st.text_area("Try it","Type your text here")
		models = ["Logistic Regression","SVM","Naive Bayes Classifier" ]

		#creating a selection for panel fo models
		choice = st.radio("Choose A Model", models)

		#Setting conditions when the user chooses "Logistic Regression"
		if choice == "Logistic Regression":
			st.info("You chose Logistic Regression")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
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

		#Setting conditions when the user chooses "SVM"
		if choice == "SVM":
			st.info("You chose Support Vector Machines")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
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

		#Setting conditions when the user chooses "Naive Bayes Classifier"
		if choice == "Naive Bayes Classifier":
			st.info("You chose Naive Bayes Classifier")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
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


		#Setting conditions when the user chooses "X"
		if choice == "X":
			st.info("You chose X")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
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
