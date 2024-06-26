# Enhancing Music Emotion Recognition with Late Fusion of High-Level Audio and Lyrics Features: A Comparative Study on the Deezer Mood Detection Dataset

This project extends previous research by investigating whether a multi-modal approach enhances music emotion recognition (MER) compared to a uni-modal approach using high-level song features and lyrics. Building on 11 song features sourced from the Spotify API, integrate lyrics features including sentiment, TF-IDF, and Anew to predict valence and arousal scores [Russel, 1980](https://psycnet.apa.org/record/1981-25062-001) on the [Deezer Mood Detection Dataset (DMDD)](https://research.deezer.com/publication/2018/09/26/ismir-delbouys.html) (Delbouys, 2018) with the incorporation of late fusion methods. Employ late fusion methods to combine these modalities, alongside four different regression models. Notably, the late fusion techniques—such as simple averaging, weighted averaging, and stacking—further improve predictive accuracy, underscoring the advantage of multi-modal features over audio features alone in predicting valence."

## Files
All the data is in the data directory, splitted into features, for music features like loudness, key, etc., lyrics, for the pure lyrics from which you can extract the sentiment via [VADER](https://github.com/cjhutto/vaderSentiment), the original deezer_dataset that comes with the valence and arousal scores which we used for the model, and a folder that holds all merged features.

The python files and respective notebooks are in the utils folder. The naming should be self-explanatory.

## Usage
We have a main.py, a finalmodel.ipynb and a lateFusion.py. You should use the lateFusion to get the results (the main.py is mainly just importing stuff).

## Citing
(https://github.com/Tibor-Krols/CogSci2-Spotify?tab=readme-ov-file)
