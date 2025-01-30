from ner_inference import get_predictions
import re
import gzip
import json

def preprocess_hindi_sentence(sentence):
    # Remove URLs
    sentence = re.sub(r'(https|http)?:\/\/\S+|www\.\S+', ' ', str(sentence))
    
    # Remove HTML tags
    sentence = re.sub(r'<.*?>', ' ', str(sentence))
    
    # Remove \n\t by space
    sentence = re.sub(r'[\n\r\t]', ' ', str(sentence))
    
    # Define Hindi characters range and additional characters to keep
    hindi_pattern = r'[\u0900-\u097F\u0020\u0964\u0965\u0966-\u096F]'
    
    # Keep only Hindi characters and specified punctuation
    sentence = ''.join(re.findall(hindi_pattern, str(sentence)))
    
    # Remove extra spaces
    sentence = re.sub(r'\s+', ' ', str(sentence)).strip()
    
    return sentence

def generate_trigrams(text, word_to_combine):
    try:
        text = preprocess_hindi_sentence(text)
        text = get_predictions(text)
        words = text.split()
        output = []
        for i in range(len(words)- word_to_combine+1):
            output.append(tuple(words[i:i+word_to_combine]))
        return output
    except TypeError:
        return None



def reading_file (file_path):
    with gzip.open(file_path, 'rt') as file:
        data = json.load(file)


    # Convert list of lists back to list of tuples
    data = [tuple(item) for item in data]
    return data


def check_trigrams(data, trigrams_to_check):
    # Filter out trigrams containing "MASK"
    filtered_trigrams_to_check = [trigram for trigram in trigrams_to_check if "MASK" not in trigram]
    
    # Determine missing trigrams
    missing_trigrams = [trigram for trigram in filtered_trigrams_to_check if trigram not in data]
    
    if missing_trigrams:
        return False, missing_trigrams
    else:
        return True, []



trigrams_to_check = generate_trigrams("MASK जी, MASK ""की ओर चले",3)
file_path = 'data.json.gz'
data_for_check = reading_file(file_path)

# Check the trigrams
result, missing_trigrams = check_trigrams(data_for_check, trigrams_to_check)

print(f'Sentence is gramatically correct: {result}')
if not result:
    print(f'Missing trigrams: {missing_trigrams}')
