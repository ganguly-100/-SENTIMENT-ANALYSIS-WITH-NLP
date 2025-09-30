#SENTIMENTANALYSIS WITH NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*:Gopaluni B N S K Ganguly

*INTERN ID*:CT08DZ1451

*DOMAIN*:MACHINE LEARING

*DURATION*:8 WEEKS

*MENTOR*:NEELA SANTHOSH

*DESCRIPTION*:

This project involved the end-to-end development of a Natural Language Processing (NLP) model designed to perform sentiment analysis on customer reviews. The primary objective was to accurately classify unstructured text data into distinct sentiment categories—positive or negative—thereby providing a valuable tool for automated feedback analysis. The technical approach was centered on using TF-IDF (Term Frequency-Inverse Document Frequency) for feature vectorization and a Logistic Regression algorithm for classification. The entire development process, from initial research to model implementation and evaluation, was facilitated by extensive use of online documentation and prototyped within the versatile Google Colab environment.

The initial phase of the project was dedicated to research and exploratory data analysis (EDA). Leveraging various online resources, including the official scikit-learn documentation and NLP-focused articles, I established a strong theoretical foundation in text preprocessing, feature extraction techniques, and classification models. Google Colab was instrumental during this stage, providing an interactive and collaborative platform to experiment with the dataset. Its notebook interface allowed for a modular, cell-by-cell approach to code execution, which was ideal for inspecting the data with the pandas library, testing different text-cleaning functions, and visualizing the data distribution. This iterative process was crucial for understanding the nuances of the text data before building the final model.

Following the exploratory phase, the core development began with a meticulous text preprocessing pipeline. This step is fundamental in any NLP task to normalize the corpus and reduce noise. The pipeline implemented several key transformations: converting all text to lowercase to ensure uniformity, and systematically removing all punctuation and numerical digits. This cleaning process standardizes the text, ensuring that the model does not treat words like "product" and "Product." as distinct entities, thereby creating a more meaningful and compact vocabulary.

The cornerstone of the model's ability to interpret text was the feature engineering stage, accomplished using the TF-IDF vectorization technique. Unlike simpler methods like word counts, TF-IDF provides a more sophisticated numerical representation of text by weighing the importance of each word. It calculates a score based on both the term frequency (how often a word appears in a review) and the inverse document frequency (how rare the word is across all reviews). This method effectively highlights keywords that are significant indicators of sentiment while down-weighting common, less informative words (e.g., "the," "a," "is"), leading to a more robust feature set for the classifier.

For the classification task, a Logistic Regression model was selected for its efficiency, interpretability, and strong performance as a baseline model for binary classification. The model was trained on the TF-IDF vectors generated from the preprocessed training data. The evaluation was conducted on an unseen test set to provide an unbiased assessment of its predictive power. Key metrics, including accuracy, a detailed classification report (with precision, recall, and F1-score), and a confusion matrix, were used to thoroughly analyze the model's performance. The resulting high accuracy and clear results from the confusion matrix demonstrated the model's effectiveness in distinguishing between positive and negative reviews, validating the entire development approach. Finally, the trained model was tested on new, unseen sentences to confirm its practical applicability in real-world scenarios.

*OUTPUT*

<img width="1920" height="1020" alt="Sentiment Analysis with TF-IDF and Logistic Regression - Colab - Google Chrome 01-10-2025 00_16_00" src="https://github.com/user-attachments/assets/038d847b-df90-43dd-962c-d8f3b8c7ec3f" />
<img width="1920" height="1020" alt="Sentiment Analysis with TF-IDF and Logistic Regression - Colab - Google Chrome 01-10-2025 00_16_13" src="https://github.com/user-attachments/assets/43331cb3-ab58-44e0-bc9a-d2fb8ef296a5" />
<img width="1920" height="1020" alt="Sentiment Analysis with TF-IDF and Logistic Regression - Colab - Google Chrome 01-10-2025 00_16_31" src="https://github.com/user-attachments/assets/7bad2b6a-b2a9-4d76-8702-9f9501752b79" />
