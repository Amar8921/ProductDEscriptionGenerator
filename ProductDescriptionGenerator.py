from flask import Flask, request, jsonify
import asyncio
import httpx
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain

# Flask app setup
app = Flask(__name__)

# Define the template for generating concise product descriptions
template = (
    "Create a concise, two-line product description for a product named '{product_name}', "
    "which belongs to the '{product_category}' category. Highlight its key benefit briefly. "
    "Ensure the description is in plain text without any emojis or special characters."
)

# Instantiate the model and the prompt template
model = OllamaLLM(model="gemma2:2b")
prompt = ChatPromptTemplate.from_template(template)

# Combine the prompt and model into a chain
chain = prompt | model

async def generate_description(product_name, product_category):
    inputs = {
        "product_name": product_name,
        "product_category": product_category
    }
    try:
        description = await asyncio.to_thread(chain.invoke, inputs)
        description = description.replace("\n", " ").replace("\r", "").strip()
        return description
    except Exception as e:
        return f"Error generating description: {str(e)}"

@app.route('/generate_description', methods=['POST'])
def generate_product_description():
    data = request.json
    product_name = data.get('product_name')
    product_category = data.get('product_category')
    
    if not product_name or not product_category:
        return jsonify({"error": "Both 'product_name' and 'product_category' are required"}), 400

    # Generate the description asynchronously
    description = asyncio.run(generate_description(product_name, product_category))
    
    # Return the result as JSON
    return jsonify({
        "product_name": product_name,
        "product_category": product_category,
        "description": description
    })

if __name__ == "__main__":
    app.run(debug=True,port=5003)
