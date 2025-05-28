continuous_train_dataset_labels = {
    'AffectiveText': {
        'labels': ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'],
        'helpers': ['valence']
        },
    'EmoInt': {
        'labels': ['anger', 'fear', 'joy', 'sadness'],
        'helpers': [],
        'fill_missing': 0.0
        },
    'REN20k': {
        'labels': ["afraid", "amused", "angry", "annoyed", "don't care", "happy", "inspired", "sad"],
        'helpers': []
        },
    'Semeval2018Intensity': {
        'labels': ['anger', 'fear', 'joy', 'sadness'],
        'helpers': ['valence']
        },
    'SentimentalLIAR': {
        'labels': ['anger', 'fear', 'joy', 'disgust', 'sad'],
        'helpers': ['sentiment', 'sentiment_score']
        }
}

discrete_single_train_dataset_labels = {
    'CARER': {
        'labels': ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'],
        'helpers': [],
        'convert_to_bool': False
        },
    'CrowdFlower': {
        'labels': ['anger', 'boredom',	'empty', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief',
                   'sadness', 'surprise', 'worry'],
        'helpers': [],
        'convert_to_bool': True
        },
    'EmotionStimulus': {
        'labels': ['anger',	'disgust', 'fear', 'happy',	'sad', 'shame',	'surprise'],
        'helpers': [],
        'convert_to_bool': False
        },
    'ISEAR': {
        'labels': ['anger',	'disgust', 'fear', 'guilt',	'joy', 'sadness', 'shame'],
        'helpers': [],
        'convert_to_bool': False
        },
    'StockEmotions': {
        'labels': ['ambiguous',	'amusement', 'anger', 'anxiety', 'belief', 'confusion',	'depression', 'disgust',
                   'excitement', 'optimism', 'panic', 'surprise'],
        'helpers': ['sentiment'],
        'convert_to_bool': False
        },
    'TEC': {
        'labels': ['anger',	'disgust', 'fear', 'joy', 'sadness', 'surprise'],
        'helpers': [],
        'convert_to_bool': False
        },
    'WASSA22': {
        'labels': ['anger',	'disgust', 'fear', 'joy', 'neutral', 'sadness',	'surprise'],
        'helpers': ['empathy', 'distress'],
        'convert_to_bool': False
        }
}

discrete_multi_train_dataset_labels = {
    'GoEmotions': {
        'labels': ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
                   'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                   'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization',
                   'relief', 'remorse', 'sadness', 'surprise'],
        'helpers': ['example_very_unclear'],
        'convert_to_bool': False
        },
    'Hurricanes8': {
        'labels': ['aggressiveness', 'awe',	'contempt',	'disapproval', 'love', 'optimism', 'remorse', 'submission'],
        'helpers': [],
        'convert_to_bool': False
        },
    'Hurricanes24': {
        'labels': ['acceptance', 'admiration', 'amazement', 'anger', 'annoyance', 'anticipation', 'apprehension',
                   'boredom', 'disgust', 'distraction', 'ecstasy', 'fear', 'grief', 'interest', 'joy', 'loathing',
                   'pensiveness', 'rage', 'sadness', 'serenity', 'surprise', 'terror', 'trust', 'vigilance'],
        'helpers': ['hurricane'],
        'convert_to_bool': True
        },
    'Semeval2018Classification': {
        'labels': ['anger',	'anticipation',	'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness',
                   'surprise', 'trust'],
        'helpers': [],
        'convert_to_bool': False
        },
    'SSEC': {
        'labels': ['anger',	'anticipation',	'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'],
        'helpers': [],
        'convert_to_bool': False
        },
    'UsVsThem': {
        'labels': ['anger',	'contempt',	'disgust', 'fear', 'gratitude', 'guilt', 'happiness', 'hope', 'pride',
                   'relief', 'sadness', 'sympathy',	'neutral'],
        'helpers': ['usVSthem_scale'],
        'convert_to_bool': False
        },
    'XED': {
        'labels': ['anger',	'anticipation',	'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'neutral'],
        'helpers': [],
        'convert_to_bool': False
        }
}

continuous_test_dataset_labels = {
    'EmoBank': {
        'labels': ['V', 'A', 'D'],
        'helpers': []
        },
    'FBValenceArousal': {
        'labels': ['valence', 'arousal'],
        'helpers': []
        }
}

discrete_single_test_dataset_labels = {
    'GoodNewsEveryone': {
        'labels': ['anger',	'annoyance', 'disgust',	'fear',	'guilt', 'joy',	'love_including_like',
                   'negative_anticipation_including_pessimism',	'negative_surprise',
                   'positive_anticipation_including_optimism', 'positive_surprise',	'pride', 'sadness',	'shame',
                   'trust'],
        'helpers': ['intensity'],
        'convert_to_bool': False
        },
    'TalesEmotions': {
        'labels': ['angry',	'disgusted', 'fearful',	'happy', 'neutral',	'positively surprised',	'negatively surprised'],
        'helpers': [],
        'convert_to_bool': True
        }
}

discrete_multi_test_dataset_labels = {
    'CancerEmo': {
        'labels': ['anger',	'anticipation',	'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'],
        'helpers': [],
        'convert_to_bool': False
        },
    'ElectoralTweets': {
        'labels': ['acceptance', 'admiration', 'amazement', 'anger or annoyance or hostility or fury',
                   'anticipation or  expectancy or interest', 'calmness or serenity', 'disappointment', 'disgust',
                   'dislike', 'fear or apprehension or panic or terror', 'hate', 'indifference',
                   'joy or happiness or elation', 'like', 'sadness or gloominess or grief or sorrow', 'surprise',
                   'trust', 'uncertainty or indecision or confusion', 'vigilance'],
        'helpers': ['annotator_trusts', 'arousal', 'valence'],
        'convert_to_bool': True
        }
}

rename_dict = {
    'amusement': 'amused',
    'anger or annoyance or hostility or fury': 'anger',
    'angry': 'anger',
    'annoyance': 'annoyed',
    'anticipation or  expectancy or interest': 'anticipation',
    'calmness or serenity': 'serenity',
    'confusion': 'confused',
    'disgusted': 'disgust',
    "don't care": 'neutral',
    'fearful': 'fear',
    'fear or apprehension or panic or terror': 'fear',
    'happiness': 'happy',
    'joy or happiness or elation': 'joy',
    'sad': 'sadness',
    'sadness or gloominess or grief or sorrow': 'sadness',
    'uncertainty or indecision or confusion': 'confused'
}

emotion_grouping_mapping = {
    'acceptance': 'trust',
    'admiration': 'love',
    'aggressiveness': 'anger',
    'afraid': 'fear',
    'amazement': 'surprise',
    'amused': 'joy',
    'anger': 'anger',
    'annoyed': 'anger',
    'anticipation': 'anticipation',
    'apprehension': 'fear',
    'approval': 'trust',
    'awe': 'surprise',
    'boredom': 'neutral',
    'caring': 'love',
    'confused': 'surprise',
    'contempt': 'disgust',
    'curiosity': 'anticipation',
    'desire': 'anticipation',
    'disappointment': 'sadness',
    'disapproval': 'disgust',
    'disgust': 'disgust',
    'dislike': 'disgust',
    'distraction': 'neutral',
    'ecstasy': 'joy',
    'embarrassment': 'shame',
    'excitement': 'joy',
    'fear': 'fear',
    'gratitude': 'trust',
    'grief': 'sadness',
    'guilt': 'shame',
    'happy': 'joy',
    'hate': 'anger',
    'hope': 'anticipation',
    'indifference': 'neutral',
    'interest': 'anticipation',
    'inspired': 'anticipation',
    'joy': 'joy',
    'like': 'love',
    'loathing': 'disgust',
    'love': 'love',
    'nervousness': 'fear',
    'neutral': 'neutral',
    'optimism': 'anticipation',
    'pensiveness': 'sadness',
    'pessimism': 'sadness',
    'pride': 'joy',
    'rage': 'anger',
    'realization': 'surprise',
    'relief': 'joy',
    'remorse': 'shame',
    'sadness': 'sadness',
    'serenity': 'joy',
    'submission': 'fear',
    'surprise': 'surprise',
    'sympathy': 'sadness',
    'terror': 'fear',
    'trust': 'trust',
    'vigilance': 'anticipation'
}

new_embedding_label_columns = list(sorted(list(set(emotion_grouping_mapping.values())), key=lambda x: x, reverse=False))

datasets_name = ['continuous_label', 'discrete_single_label', 'discrete_multi_label']

yes_no_to_bool_mapper = {'yes': True, 'no': False}
