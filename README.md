# 🌳Agri-GPT🌳

Agri-GPT is designed to answer queries about quantifying and reporting emissions in the agricultural industry, based on agricultural guidance documents published under the GHG Protocol.

The problem: 
* It is crucial to quantify and measure emissions for the agri industry to guide reduction strategies because the agricultural industry accounts for ~20% of global emissions.
* Yet, accounting for emissions in this sector is complex due to the diversity of emission sources and measurement techniques.
* It is tedious to comb through extensive sector-specific guidance documents.

The Solution: 
* A RAG-based app for an Agricultural guidance document published by the GHG protocol.

## Usage

Visit [https://agri-gpt.streamlit.app](https://agri-gpt.streamlit.app)

## How It Works

1. **Document Loading and Preprocessing**: Initially, agricultural guidance documents are loaded and preprocessed. This includes cleaning the text and splitting it into smaller chunks for easier processing.
2. **Embedding and Retrieval**: Document chunks are embedded into a vectorstore for retrieval based on user queries.
3. **User Query**: Users input their queries related to agricultural emissions which are then processed by the application.
4. **Response Generation**: The application utilizes OpenAI's GPT models, alongside custom retrieval and processing pipelines to generate relevant responses to the user's query.


