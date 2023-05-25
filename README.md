
# FURI
### By Nicolas Burton
### This is a project for the Fulton Undergraduate Research Initiative at Arizona State University.

<p>This project aims to predict the public’s opinion on any given pop music using a machine learning model and the music’s audio features. While other studies have been shown to predict public ratings of pop music, those studies only used the metadata, such as the music’s artist and album. By using audio features instead of only metadata to predict public opinions on music, this project will help musical artists without much previous musical data to understand how their pop music will be received and allow them to improve their music before they publish it.</p>

<p>If you want to try out the code for yourself, you must download ffmpeg along with the other imported modules to convert the mp3 files to wav files. You must also get your own Spotify client ID and secret to use the Spotify API. The code will print out instructions on the first two runs in order to get the data all setup.</p>

### Data of the tracks

<p align="center">
  <img src="./images/Screenshot%202023-05-24%20174856.png?raw=true">
</p>

<p>Spotify would regularly update their music so I ended up using this set of music from March 9, 2023 for the rest of the project.</p><br>
<p>A Gaussian Mixture Model is used to split the tracks into groups based on the AIC and BIC scores</p>

<p align="center">
  <img src="./images/Screenshot%202023-05-24%20175005.png?raw=true">
</p>

<p>Here a split into two, three, or four groups can be done.<br> I went with a split of two groups as shown below.</p>

<p align="center">
  <img src="./images/Screenshot%202023-05-24%20175058.png?raw=true">
</p>


### Features of the tracks

<p>This is the wave form of the track called 2Step. The next three features come from 2Step as well. By extracting certain features from the track, I could reduce the amount the learning that the model will have to do and push the model in a certain direction.</p>
<p align="center">
  <img src="./images/Screenshot%202023-05-24%20175232.png?raw=true">
</p>

<p>This is the Mel spectrogram of 2Step. This feature has the most information out of all three features, containing perceptually-relevant amplitude representation.</p>
<p align="center">
  <img src="./images/Screenshot%202023-05-24%20175603.png?raw=true">
</p>

<p>This is the Mel-frequency cepstral coefficient of 2Step. This seems like a usless feature due to, but it is used to identify the vocals of the track. The vocals can be identified by the sharp wave structue present in this feature.</p>
<p align="center">
  <img src="./images/Screenshot%202023-05-24%20175548.png?raw=true">
</p>

<p>This is the Chromagram of 2Step. This feature extracts the pitch being played into 12 pitch classes.</p>
<p align="center">
  <img src="./images/Screenshot%202023-05-24%20175455.png?raw=true">
</p><br>

### Results of the model

<p>These are the results of the model. The model had low loss that increased after epoch 25 due to overfiting. The model accuracy was <b>31.25%</b> at epoch 25.</p>
<p align="center">
  <img src="./images/Screenshot%202023-05-24%20175629.png?raw=true">
</p>

<p align="center">
  <img src="./images/Screenshot%202023-05-24%20175644.png?raw=true">
</p>
