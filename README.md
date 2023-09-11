# AI-integrated-video-editor

## 1. Set up a web interface:
   Create a web interface using Flask. Flask is a web framework for building web applications in Python.
   Define route for homepage and another one for video uploading.
   Create and render HTML & CSS templates and run the flask app.

## 2. Upload and store videos:
   Create an HTML Form for Video Upload.
   Handle the Video Upload in Flask Route.

   Store the uploaded videos on application server's local file system or utilize cloud-based object storage services like Amazon S3, Google Cloud Storage, or Azure Blob Storage.

## 3. Create a metadata file:
   Metadata files are files that store additional information about the associated data or objects.
   For this we have to specify the metadata fields like the title, the date, the author, and the keywords,etc.
   Extract metadata from videos using AI tools like Google Cloud Vision API, Amazon Rekognition.
   Save the file in the CSV, JSON, or XML formats.

## 4. Video Processing Pipeline:
   We can create a pipeline to carry out the following steps:
   
   ### a. Generating Script -
   We can use Automatic Speech Recognition (ASR) models, such as Google's Speech-to-Text API for automatically generating script out of videos.
   Automatic Speech Recognition (ASR) is a technology that is designed to convert spoken language into text. They are trained on large datasets of spoken language and have
   state-of-the-art accuracy in transcribing audio to text, making them suitable for generating a script from spoken words in videos.
   
   Google's Speech-to-Text API is a service provided by Google Cloud which can transcribe spoken words from various audio sources, such as audio recordings, live streaming,
   or telephony conversations. We can follow the following steps in order to use Google's Speech-to-Text API for generating script out of videos.
     
   #### Sign in to Google Cloud console
   
   #### Go to the project selector page
   
   #### We can either choose an existing project or create a new one
   
  #### If we create a new project, we will be prompted to link a billing account to this project. If we are using a pre-existing project, we need to make sure that the billing is enabled.
   
  #### To try Speech-to-Text without linking it to our project, we can choose the "TRY THIS API" option. To enable the Speech-to-Text API for use with project, we can click ENABLE.
   
  #### (Optional) Enable data logging. By opting in to data logging, we allow Google to record any audio data that we send to Speech-to-Text. This data is used to improve the Speech-to-Text models.
  #### Users who opt in to data logging benefit from lower pricing.
   
  #### We now must link one or more service accounts to the Speech-to-Text API. Click on the Credentials menu item on the left side of the Speech-to-Text API main page. If we do not have any service
  #### accounts associated with this project, we can create one by following the instructions in the "creating a new service account" section.

   ### b. Name Entity Recognition  -
   Named Entity Recognition (NER) is a natural language processing (NLP) technique that focuses on identifying and classifying named entities (i.e.,names, specific objects, places, organizations, dates, 
   quantities, and other proper nouns) within a text.
   For this purpose we can use model like spaCy for recognizing entities in the script.

   #### Install spaCy and Download Language Model:
   Install spaCy and download a language model. spaCy provides pre-trained models for multiple languages. For English, we can download the en_core_web_sm model, which is a small English language model.

   #### Load the Language Model:
   Once we've downloaded the language model, load it using spaCy.

   #### Tokenize and Analyze Text:
   Use the loaded language model to tokenize and analyze text. 

   #### Access Named Entities:
   Access the named entities found in the text, along with their respective labels

   #### Visualizing NER:
   Visualise the name entities and their respective labels using displacy library.
   
   ### c. Sentiment Analysis -
   For sentiment analysis we can make use of GloVe (Global Vectors for Word Representation), which is a word embedding technique used in natural language processing (NLP) and machine learning. GloVe vectors
   capture semantic relationships between words based on their co-occurrence patterns in large text corpora. This means that words with similar meanings have similar vector representations. 

   To use GloVe vectors for sentiment analysis, we can follow these steps:
  
   #### Download GloVe Pre-trained Word Vectors:
   First, we need to download pre-trained GloVe word vectors. These vectors come in various dimensions (e.g., 50, 100, 200, or 300 dimensions). We can choose the one that suits our needs from the official 
   GloVe website (https://nlp.stanford.edu/projects/glove/).

   #### Tokenize and Preprocess Text:
   We can use libraries like nltk or spaCy for tokenization and cleaning. Tokenization involves - Punctuation Handling, Stop Word Removal, Removal of hyperlinks, Stemming and Lemmatization, and much more.
   
   #### Load GloVe Word Vectors:
   We can load glove word vectors to convert word into vectors.

   #### Convert text into Vectors:
   We can define a function which converts text into vectors by using numpy library. 

   #### Train the model:
   Train the model using different machine learning algorithms. Tests different algorithms and select the one providing for highest accuracy.

   #### Predict Sentiment:
   Use the saved model to make the predictions on unseen data.
   
   
   Why NER & Sentiment Analysis is required- These are required for content categorization, sentiment-based editing, and script analysis.

   ### d. Facial Detection-
   A facial detection model is integrated to identify and mark timestamps when faces are present in the video. OpenCV's pre-trained Haar Cascades, deep learning-based models like Single Shot MultiBox 
   Detector (SSD), or Region-based Convolutional Neural Networks (R-CNNs) can be used for
   detecting faces in the video.
   Haar Cascade Classifiers, are a machine learning object detection method used to identify objects, detect faces or patterns within images or video. They are trained using machine
   learning techniques, particularly the AdaBoost (Adaptive Boosting) algorithm.
   SSD is a single-shot object detection model that efficiently detects objects of various sizes and shapes in real-time and also provide information about their location (bounding
   boxes) and class labels (e.g., car, person, dog).
   R-CNN, or Region-based Convolutional Neural Network, is an object detection method in computer vision that uses a combination of region proposals and convolutional neural networks
   to identify and locate objects within an image.

   For this particular product, we can make use of Haar Cascade Classifiers. Here's the piece of code that we can use for detecting faces in videos:
   
   ####  Importing the OpenCV library which is used for computer vision tasks like image and video processing.
   import cv2

  ####  We can create an instance of the CascadeClassifier class from OpenCV, which is pre-trained for face detection using Haar cascades. It loads the XML file containing the trained
   ####  model for frontal face detection.
   facescascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
   
   #### Here, a VideoCapture object is created to open and read frames from the video file
   cap = cv2.VideoCapture('file_name.mp4')
   
   ####  We can start an infinite loop to continuously process video frames until the user decides to exit.
   while True:
   #### Read a frame from the video
   success, img = cap.read()
    
   #### Convert the frame to grayscale for face detection. Face detection is often performed on grayscale images because it simplifies processing.
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   #### Detects faces in the grayscale frame using the detectMultiScale method of the facescascade classifier. The parameters 1.1 and 4 control the sensitivity and accuracy of detection.
   faces = facescascade.detectMultiScale(gray, 1.1, 4)
    
   #### Starts a loop to iterate through the detected faces. For each detected face, it provides the coordinates (x, y) of the top-left corner of the bounding box and the width (w) and ###height (h) of the bounding box.
   for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
   #### Display the frame with detected faces
   cv2.imshow('Video', img)
    
   #### Wait for a key press and check if the user pressed 'Esc'. If the 'Esc' key is pressed, the loop is terminated, and the program exits.
   k = cv2.waitKey(1) & 0xff
   if k == 27:
      break

   #### Release the video capture object and close the OpenCV window
   cap.release()
   cv2.destroyAllWindows()

   ### e.Video editing -
   For video editing tasks like cropping, concatenation, and special effects, we can use FFmpeg. FFmpeg is a powerful and widely used multimedia framework with a range of video processing capabilities. It's
   highly efficient and well-suited for batch video editing tasks.

 
## 6. Integrate all models and Deploy - 
Integrate all the models and deploy it on the cloud plaforms like AWS, Heroku or Azure.


