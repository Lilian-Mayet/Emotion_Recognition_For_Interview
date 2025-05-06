import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np

def analyze_dataset(data_directory):
    """Analyze the speech emotion dataset and generate visualizations using matplotlib."""

    # Initialize data structures
    data = {
        'filename': [],
        'emotion': [],
        'gender': [],
        'duration': [],
        'intensity': [],
        'zero_crossing_rate': [],
        'sample_rate': []
    }

    emotion_mapping = {
        "SAD": "sadness",
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral"
    }

    print("Analyzing dataset...")
    for subdir, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(subdir, file)

                emotion_code = file.split("_")[2]
                emotion = emotion_mapping.get(emotion_code, "unknown")
                gender = file.split("_")[3]

                try:
                    y, sr = librosa.load(file_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                    intensity = np.mean(np.abs(y))

                    data['filename'].append(file)
                    data['emotion'].append(emotion)
                    data['gender'].append(gender)
                    data['duration'].append(duration)
                    data['intensity'].append(intensity)
                    data['zero_crossing_rate'].append(zcr)
                    data['sample_rate'].append(sr)

                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

    df = pd.DataFrame(data)

    # Set up the plots

    # 1. Emotion Distribution
    plt.figure(figsize=(12, 6))
    emotion_counts = df['emotion'].value_counts()
    plt.bar(emotion_counts.index, emotion_counts.values)
    plt.title('Distribution of Emotions in Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Gender Distribution
    plt.figure(figsize=(8, 8))
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('Gender Distribution')
    plt.show()

    # 3. Duration Distribution by Emotion
    plt.figure(figsize=(12, 6))
    emotions = sorted(df['emotion'].unique())
    box_data = [df[df['emotion'] == emotion]['duration'] for emotion in emotions]
    plt.boxplot(box_data, labels=emotions)
    plt.title('Duration Distribution by Emotion')
    plt.xlabel('Emotion')
    plt.ylabel('Duration (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. Intensity vs Zero Crossing Rate
    plt.figure(figsize=(10, 6))
    for emotion in emotions:
        mask = df['emotion'] == emotion
        plt.scatter(df[mask]['intensity'], 
                   df[mask]['zero_crossing_rate'], 
                   label=emotion, 
                   alpha=0.6)
    plt.title('Intensity vs Zero Crossing Rate by Emotion')
    plt.xlabel('Intensity')
    plt.ylabel('Zero Crossing Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. Duration Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['duration'], bins=30, edgecolor='black')
    plt.title('Distribution of Audio Duration')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # 6. Emotion Distribution by Gender
    plt.figure(figsize=(12, 6))
    df_grouped = df.groupby(['emotion', 'gender']).size().unstack()
    df_grouped.plot(kind='bar', width=0.8)
    plt.title('Emotion Distribution by Gender')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.legend(title='Gender')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 7. Sample Rate Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['sample_rate'], bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Sample Rates')
    plt.xlabel('Sample Rate (Hz)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_directory = "speech_feelings/data"
    analyze_dataset(data_directory)
