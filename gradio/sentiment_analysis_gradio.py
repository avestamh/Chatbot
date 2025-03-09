import gradio as gr
from transformers import pipeline

# Load pre-trained sentiment-analysis from Hugging Face
'''The pipeline("sentiment-analysis") from Hugging Face loads a pre-trained model
 if it is defult it typically is distilbert-base-uncased-finetuned-sst-2-english, 
 which is fine-tuned for sentiment analysis tasks. 
 It returns a label (POSITIVE or NEGATIVE) along with a confidence score.'''

# Specify device=0 to use the first CUDA device (GPU)

# sentiment_analyzer = pipeline('sentiment-analysis')
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

# Define a function that performs sentiment analysis
def analyze_sentiment(text):
    # Use the sentiment analyzer to predict sentiment
    result = sentiment_analyzer(text)
    label = result[0]['label']
    confidence = result[0]['score']
    return f"Sentiment: {label}, Confidence: {confidence:.2f}"

# Create an interface
'''gr.Interface object where we can specify the function (fn), input type (inputs), 
and output type (outputs)'''

# Create an interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter a text"),
    outputs="text"
)

# Launch the interface
iface.launch()
## iface.launch(share=True) if you want to make it accessible for public
