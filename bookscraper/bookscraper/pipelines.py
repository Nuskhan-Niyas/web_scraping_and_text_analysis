# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


#import summy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from itemadapter import ItemAdapter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transformers import pipeline
#from summy import summarize


class BookscraperPipeline:
    
    def __init__(self):
        # Initialize Sentiment Analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize KeyBERT with all-MiniLM-L6-v2
        self.kb = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
        
        # Load T5 Summarization Model (Small version for efficiency)
        self.summarizer = pipeline("summarization", model="t5-small")
        
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        ############################## Data Cleaning ##############################################################
        # Strip all white spaces from strings
        field_names = adapter.field_names()
        for field_name in field_names:
            value = adapter.get(field_name)
            
            if isinstance(value, tuple):
                value = value[0]  # Extract the first element if it's a tuple
            
            if isinstance(value, str):  # Proceed only if value is a string
                adapter[field_name] = value.strip()
        
        # Category and product type -> switch to lowercase
        lowercase_keys = ['category', 'product_type']
        for lowercase_key in lowercase_keys:
            value = adapter.get(lowercase_key)
            if value:
                adapter[lowercase_key] = value.lower()

        # Price -> convert to float 
        price_keys = ['price', 'price_excl_tax', 'price_incl_tax', 'tax']
        for price_key in price_keys:
            value = adapter.get(price_key)
            if value:
                value = value.replace('£', '').strip()
                try:
                    adapter[price_key] = float(value)
                except ValueError:
                    adapter[price_key] = 0.0  # Fallback in case of invalid value

        # Availability -> extract number of books in stock
        availability_string = adapter.get('availability', '')
        split_string_array = availability_string.split('(')
        if len(split_string_array) < 2:
            adapter['availability'] = 0
        else:
            availability_array = split_string_array[1].split(' ')
            try:
                adapter['availability'] = int(availability_array[0])
            except ValueError:
                adapter['availability'] = 0

        # Reviews -> convert string to number
        num_reviews_string = adapter.get('num_reviews', '0')
        try:
            adapter['num_reviews'] = int(num_reviews_string)
        except ValueError:
            adapter['num_reviews'] = 0

        # Stars -> convert text to number
        stars_string = adapter.get('stars', '')
        if stars_string:
            split_stars_array = stars_string.split(' ')
            stars_text_value = split_stars_array[1].lower() if len(split_stars_array) > 1 else ""
            stars_map = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5
            }
            adapter['stars'] = stars_map.get(stars_text_value, 0)
        
        ############################## Sentiment Analysis ##########################################################
        description = adapter.get('description', '')
        if description:
            sentiment_score = self.sia.polarity_scores(description)['compound']  # Get sentiment score

            # Determine sentiment direction
            if sentiment_score > 0.05:
                sentiment_label = "Positive"
            elif sentiment_score < -0.05:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
        else:
            sentiment_score = 0
            sentiment_label = "Neutral"

        # Store sentiment results in the item
        adapter['sentiment_score'] = round(sentiment_score, 2)
        adapter['sentiment_label'] = sentiment_label
        
         ############################## Summarization ##############################################################
        # Summarize the description to extract key insights
        #if description:
         #   summary = summarize(description, num_sentences=2)  # Limit to 2 sentences for summary
          #  adapter['summary'] = summary
        
        ############################## Keyphrase Extraction (Find Common Themes) ##############################################################
        if description:
            keyphrases = self.kb.extract_keywords(description, keyphrase_ngram_range=(1, 2), top_n=5)
            adapter['keyphrases'] = [phrase[0] for phrase in keyphrases]  # Extract only the keyphrases (without scores)

        ############################## T5 Summarization ##########################################################
        if description and len(description) > 20:  # Only summarize if text is long enough
            summary = self.summarizer(description, max_length=50, min_length=10, do_sample=False)
            adapter['summary'] = summary[0]['summary_text']
        else:
            adapter['summary'] = description  # If too short, keep original

        return item

        










