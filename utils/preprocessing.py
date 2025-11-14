"""
Preprocessing utilities for emotion detection
Contains all text cleaning and preprocessing functions
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
def download_nltk_resources():
    """Download all required NLTK resources"""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

# Initialize NLTK resources
download_nltk_resources()

class TextPreprocessor:
    """Main text preprocessing class for emotion detection"""
    
    def __init__(self, remove_stopwords=True, lemmatize=True, 
                 remove_punctuation=True, lowercase=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Get English stopwords but keep emotion words
        self.stop_words = set(stopwords.words('english'))
        emotion_words = {'happy', 'sad', 'angry', 'fear', 'love', 'surprise',
                        'joy', 'hate', 'excited', 'worried', 'scared', 'terrified',
                        'wonderful', 'terrible', 'amazing', 'awful'}
        self.stop_words = self.stop_words - emotion_words
    
    def clean_text(self, text):
        """Master function to clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Apply preprocessing steps
        text = self.expand_contractions(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_html_tags(text)
        text = self.remove_special_characters(text)
        text = self.remove_numbers(text)
        
        if self.lowercase:
            text = text.lower()
        
        # Tokenization
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Lemmatization
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Remove punctuation
        if self.remove_punctuation:
            tokens = self.remove_punct(tokens)
        
        # Remove short words
        tokens = [t for t in tokens if len(t) > 1]
        
        # Join tokens back to string
        cleaned_text = ' '.join(tokens)
        
        # Remove extra whitespaces
        cleaned_text = self.remove_extra_whitespace(cleaned_text)
        
        return cleaned_text
    
    def expand_contractions(self, text):
        """Expand contractions like don't -> do not"""
        contractions_dict = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it's": "it is", "let's": "let us", "shouldn't": "should not",
            "that's": "that is", "there's": "there is", "they're": "they are",
            "wasn't": "was not", "we're": "we are", "weren't": "were not",
            "what's": "what is", "won't": "will not", "wouldn't": "would not",
            "you're": "you are", "you've": "you have"
        }
        
        for contraction, expansion in contractions_dict.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    
    def remove_emails(self, text):
        """Remove email addresses from text"""
        return re.sub(r'\S+@\S+', '', text)
    
    def remove_html_tags(self, text):
        """Remove HTML tags from text"""
        return re.sub(r'<.*?>', '', text)
    
    def remove_special_characters(self, text):
        """Remove special characters except alphanumeric and spaces"""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def remove_numbers(self, text):
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)
    
    def tokenize(self, text):
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except:
            return text.split()
    
    def remove_stop_words(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens to their base form"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def remove_punct(self, tokens):
        """Remove punctuation from tokens"""
        return [token for token in tokens if token not in string.punctuation]
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespaces from text"""
        return ' '.join(text.split())


def quick_clean(text):
    """Quick cleaning function with default settings"""
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text)


def batch_clean(texts):
    """Clean multiple texts at once"""
    preprocessor = TextPreprocessor()
    return [preprocessor.clean_text(text) for text in texts]


if __name__ == "__main__":
    test_text = "I'm so EXCITED!!! This is AMAZING! I can't believe it's happening!!!"
    print("Original:", test_text)
    print("Cleaned:", quick_clean(test_text))
