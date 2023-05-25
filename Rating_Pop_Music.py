
#Setup
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.mixture import GaussianMixture
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ['SPOTIPY_CLIENT_ID'] = 'your client ID here'
os.environ['SPOTIPY_CLIENT_SECRET'] = 'your client secret here'

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)

def main():
    if not os.path.exists("./Npy_Values"):
        # Run setupTracks() once and never again since Spotify updates their music over time
        if not os.path.exists("pop_tracks.xlsx"):
            print('Creating spreadsheet of tracks')
            setupTracks()
            if os.path.exists("./Tracks"):
                print('Outdated Tracks in ./Tracks directory')
                print('Delete the Directory and its contents using this command line:')
                print('rmdir ./Tracks')
                print('Re-run this python file')
                return

        if not os.path.exists("./Tracks"):
            print('Create a directory called "Tracks" using this command line:')
            print('mkdir Tracks')
            print('Change directory to "Tracks" using this command line:')
            print('cd ./Tracks')
            print('Convert all mp3 previews to wav using this command line:')
            print('for f in *.mp3; do ffmpeg -i "$f" "${f%.mp3}.wav"; done')
            print('Remove all mp3 files using this command line:')
            print('rm *.mp3')

            print('Change directory back to original directory using this command line:')
            print('cd ..')
            print('Re-run this python file')
            return

        #Load and Plot data from xlsx file
        xlsx_track = pd.read_excel('pop_tracks.xlsx')
        xlsx_track.drop(columns='Unnamed: 0',inplace=True)  # Idk how this column gets added, but we drop it
        xlsx_track.fillna({'preview_url':''},inplace=True)

        print(f"Mean: {xlsx_track['popularity'].mean()}\nMedian: {xlsx_track['popularity'].median()}")
        plt.figure(figsize=(16, 6))
        plot = sns.displot(xlsx_track['popularity'], kde= True,alpha= 0.2, color='g')
        plot.set_axis_labels('popularity', 'Count')
        plt.title('Original Pop Music Dataset')
        plt.show()

        # Split the track data into groups using gaussian mixture model
        n_components = np.arange(1, 16)
        models = [GaussianMixture(n,covariance_type='full',random_state=42).fit(xlsx_track[['popularity']]) for n in n_components]
        gmm_model_comparisons=pd.DataFrame({'n_components' : n_components,'BIC' : [m.bic(xlsx_track[['popularity']]) for m in models],'AIC' : [m.aic(xlsx_track[['popularity']]) for m in models]})
        sns.lineplot(data=gmm_model_comparisons[['BIC','AIC']])
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Gaussian mixture model comparisons')
        plt.show()

        # Use the best splits depending on the gaussian mixture model BIC and AIC scores
        # 2 split groups for example
        print("Splitting into groups")
        xlsx_track = splitGroups(2,xlsx_track)
        if xlsx_track.empty: return

        # Of that split the upper spilt is used for the rest of this project
        df_in = xlsx_track.loc[xlsx_track['pred_cluster']==0]
        df_in.reset_index(inplace=True)

        print("Getting features")
        mel_mfcc_features,chroma_features, labels = get_features(df_in)
        getTrainTestVal(mel_mfcc_features, chroma_features, labels)

    else:
        print("Directory ./Npy_Values already exists")
        print("Skipping to model creation")

    ModelAndResult()


def normalize(arr):
    arr = (arr-arr.min())/(arr.max()-arr.min())
    return arr

# Make the model and result
def ModelAndResult():
    X1_test= np.load('Npy_Values/X1_test.npy')
    X1_train= np.load('Npy_Values/X1_train.npy')
    X1_val= np.load('Npy_Values/X1_val.npy')
    X2_test= np.load('Npy_Values/X2_test.npy')
    X2_train= np.load('Npy_Values/X2_train.npy')
    X2_val= np.load('Npy_Values/X2_val.npy')
    y_test= np.load('Npy_Values/y_test.npy')
    y_train= np.load('Npy_Values/y_train.npy')
    y_val= np.load('Npy_Values/y_val.npy')

    # Normalize values
    X1_test = normalize(X1_test)
    X1_train = normalize(X1_train)
    X1_val = normalize(X1_val)

    X2_test = normalize(X2_test)
    X2_train = normalize(X2_train)
    X2_val = normalize(X2_val)

    X2_test=X2_test[...,np.newaxis]
    X2_train=X2_train[...,np.newaxis]
    X2_val=X2_val[...,np.newaxis]

    # Reduce y values to a range between 0 and 7
    y_test = np.round(y_test/5)-13
    y_train = np.round(y_train/5)-13
    y_val = np.round(y_val/5)-13

    model = build_model((128,1280,2),(12,1280,1))

    opt = tf.keras.optimizers.Adam(weight_decay=.01,learning_rate=0.0005)

    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    model.summary()

    history = model.fit(x=[X1_train,X2_train], y=y_train,
                    validation_data=([X1_val,X2_val], y_val), 
                    batch_size=16, 
                    epochs=50)
    
    plot_history(history)
    test_loss, test_acc = model.evaluate(x=[X1_test,X2_test], y=y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    print(f'Test loss: {test_loss}')


# Build the model
def build_model(mel_mfcc_shape, chroma_shape):
    input_mel_mfcc = tf.keras.Input(shape=mel_mfcc_shape,name="Mel_MFCC")
    input_chroma = tf.keras.Input(shape=chroma_shape,name="Chromagram")

    # Mel and mfcc layer
    features_mel_mfcc = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.MaxPooling2D((3,3), strides=(3,3), padding='same')(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.BatchNormalization()(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.SpatialDropout2D(0.7)(features_mel_mfcc)

    features_mel_mfcc = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.MaxPooling2D((3,3), strides=(3,3), padding='same')(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.BatchNormalization()(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.SpatialDropout2D(0.7)(features_mel_mfcc)

    features_mel_mfcc = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.MaxPooling2D((3,3), strides=(3,3), padding='same')(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.BatchNormalization()(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.SpatialDropout2D(0.7)(features_mel_mfcc)

    features_mel_mfcc = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.MaxPooling2D((3,3), strides=(3,3), padding='same')(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.BatchNormalization()(features_mel_mfcc)
    features_mel_mfcc = tf.keras.layers.SpatialDropout2D(0.7)(features_mel_mfcc)

    features_mel_mfcc = tf.keras.layers.Flatten()(features_mel_mfcc)

    # Chroma layer
    features_chroma = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(input_chroma)
    features_chroma = tf.keras.layers.MaxPooling2D((3,3), strides=(3,3), padding='same')(features_chroma)
    features_chroma = tf.keras.layers.BatchNormalization()(features_chroma)
    features_chroma = tf.keras.layers.SpatialDropout2D(0.7)(features_chroma)

    features_chroma = tf.keras.layers.Flatten()(features_chroma)

    # Concatenate the layers
    concat = tf.keras.layers.concatenate([features_mel_mfcc,features_chroma])

    pred = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.L1L2(0.01))(concat)
    pred = tf.keras.layers.Dropout(0.6)(pred)
    pred = tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.L1L2(0.01))(concat)
    pred = tf.keras.layers.Dropout(0.6)(pred)
    pred = tf.keras.layers.Dense(8, activation='softmax')(pred)

    model_result = tf.keras.Model(inputs=[input_mel_mfcc,input_chroma],outputs=[pred])
    return model_result


# Plot the result
def plot_history(history):
    # Create loss plot
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,51)
    plt.figure(figsize=(16,5))
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Create accuracy plot
    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    epochs = range(1,51)
    plt.figure(figsize=(16,5))
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Get X and y train, test, and validation
def getTrainTestVal(mel_mfcc_features, chroma_features, labels):
    X1_train,X1_test,X2_train,X2_test,y_train,y_test = train_test_split(mel_mfcc_features,chroma_features,labels,test_size=0.2, random_state=42)
    X1_train,X1_val,X2_train,X2_val,y_train,y_val = train_test_split(X1_train,X2_train,y_train,test_size=0.25, random_state=42)

    # Save these values because they take a long time to get again 
    os.makedirs('Npy_Values')
    np.save('Npy_Values/X1_test.npy', X1_test)
    np.save('Npy_Values/X1_train.npy', X1_train)
    np.save('Npy_Values/X1_val.npy', X1_val)
    np.save('Npy_Values/X2_test.npy', X2_test)
    np.save('Npy_Values/X2_train.npy', X2_train)
    np.save('Npy_Values/X2_val.npy', X2_val)
    np.save('Npy_Values/y_test.npy', y_test)
    np.save('Npy_Values/y_train.npy', y_train)
    np.save('Npy_Values/y_val.npy', y_val)


def generate_mel_mfcc(y_cut):
    max_size = 1280 # max audio file feature width
    n_fft = 2048
    hop_length = 512
    melspec = padding(librosa.feature.melspectrogram(y_cut, n_fft=n_fft, hop_length=hop_length, n_mels=128), 128, max_size)
    log_melspec = librosa.power_to_db(melspec)
    MFCCs = padding(librosa.feature.mfcc(y_cut, n_fft=n_fft, hop_length=hop_length, n_mfcc=128), 128, max_size)
    image = np.array(log_melspec)
    image = np.dstack((image,MFCCs))
    return image


def generate_chroma(y_cut, sr):
    max_size = 1280 #max audio file feature width
    chroma_stft = padding(librosa.feature.chroma_stft(y=y_cut, sr=sr),12,max_size)
    image = np.array(chroma_stft)
    return image


def get_features(df):   
    mel_mfcc_features = []
    chroma_features = []
    labels = []

    #For each song, determine how many augmentations are needed
    for i in range(0,df.shape[0]):
        name = df['name'][i]
        popularity = df['popularity'][i]
        url = df['preview_url'][i]
        if url == '':
            continue
        file_name = "".join(x for x in name if x.isalnum())
        saved_song = './Tracks/' + file_name + '.wav'
        y, sr = librosa.load(f'{saved_song}')
        y,_ = librosa.effects.trim(y)
        #generate features & output numpy array          
        data1 = generate_mel_mfcc(y)
        data2 = generate_chroma(y, sr)
        mel_mfcc_features.append(data1[np.newaxis,...])
        chroma_features.append(data2[np.newaxis,...])    
        labels.append(popularity)     
    output1 = np.concatenate(mel_mfcc_features,axis=0)
    output2 = np.concatenate(chroma_features,axis=0)       
    return(np.array(output1),np.array(output2),np.array(labels))


# Padding function adapted from Nicolas Gervais on https://stackoverflow.com/questions/59241216/padding-numpy-arrays-to-a-specific-size
def padding(array, xx, yy):

    #:param array: numpy array
    #:param xx: desired height
    #:param yy: desired width
    #:return: padded array

    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2,0)
    aa = max(0,xx - a - h)
    b = max(0,(yy - w) // 2)
    bb = max(yy - b - w,0)
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


# Categorize artists with their tracks
# Unused for this project
def getArtist(df):
    artist_data = {'name':[],'id':[],'track_data':[]}
    maxSpeed = 50
    while df.shape[0]%maxSpeed:
        maxSpeed-=1

    for page in range(int(df.shape[0]/maxSpeed)):
        page *= maxSpeed
        tracks = sp.tracks(df['id'][page:page+maxSpeed])
        for track in tracks['tracks']:
            arr_artist = track['artists']
            for artist in arr_artist:
                if(artist_data['id'].count(artist['id']) == 1):
                    index=artist_data['id'].index(artist['id'])
                    artist_data['track_data'][index] += ',' + track['id'] + '_' + str(track['popularity'])
                else:
                    artist_data['name'].append(artist['name'])
                    artist_data['id'].append(artist['id'])
                    artist_data['track_data'].append(track['id'] + '_' + str(track['popularity']))

    df_artist = pd.DataFrame(artist_data)
    df_artist.to_excel('artist.xlsx')


# Create groups and plot the splits
def splitGroups(n,df):
    if n not in range(1,10):
        print("Unsupported number of splits")
        return pd.DataFrame()
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42).fit(df[['popularity']])
    labels = gmm.predict(df[['popularity']])
    df['pred_cluster']=labels

    print('means of all groups:')
    for mean in enumerate(gmm.means_):
        print(f"group {mean[0]}: {mean[1][0]}")

    sns.displot(data=df, x='popularity', hue='pred_cluster', palette=sns.color_palette(palette='bright',n_colors=(n)))
    plt.title(f'Pop tracks divided into {n} groups')
    plt.show()
    return df


# Get and clean data then place in xlsx file
def setupTracks():
    df_track = getTracksInfo()

    df_track.drop_duplicates(subset=['id'],inplace=True,ignore_index=True)
    df_track.drop_duplicates(subset=['name'],inplace=True,ignore_index=True)
    df_track.fillna({'preview_url':''},inplace=True)
    df_track.to_excel('pop_tracks.xlsx')

    # Download all previews available as mp3
    for i in range(0,df_track.shape[0]):
        name = df_track['name'][i]
        url = df_track['preview_url'][i]
        if url == '':
            continue
        file_name = "".join(x for x in name if x.isalnum())
        saving_directory = 'Tracks/' + file_name + '.mp3'
        song = requests.get(url=url, stream=True)
        with open(saving_directory, 'wb') as f:
            f.write(song.content)


# Grab name, id, and popularity data from Dance Pop genre of Spotify API
def getTracksInfo():
    searching = sp.search(q='genre:pop',type='track',limit=50)

    i=50
    track_data = {'name':[],'id':[],'popularity':[],'duration_ms':[],'preview_url':[]}

    while searching:
        for track in searching['tracks']['items']:
            if track['popularity'] == 0:    # Ignore tracks that have not been listened to 
                continue
            for key in track_data:
                track_data[key].append(track[key])
        if searching['tracks']['next']:
            searching = sp.search(q='genre:pop',type='track',limit=50,offset=i)
        else:
            searching = None
        i+=50

    return pd.DataFrame(track_data)


if __name__ == "__main__":
    main()