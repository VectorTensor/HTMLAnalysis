from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from HelperMethods import * 

class TextInput(BaseModel):
    text: str

# Create an instance of the FastAPI class
app = FastAPI()

# Define a POST endpoint
@app.post("/items/")
async def create_item(input_data: TextInput):
    # Here you can process the received item, for now, just return it
    
    soup = ParseHTMLString(input_data.text)
    chunks = SplitDocumentsByTags(soup)
    embeddings = GetEmbeddings(chunks)
    dim = embeddings[0].embedding.shape[0]
    planes = 10
    buckets = 2**planes
    random_planes_matrix = np.random.normal(size=(planes, dim))
    embeddings_space = createEmbeddingSpace(embeddings, random_planes_matrix)
    similar_text = SimmilarText('something <p class=\"price\">$12.39</p> <p class=\"description\">This item is very good and of various sizes.</p>',embeddings, random_planes_matrix,0.8)
    products_info=[]
   # print(similar_text)
    for i in similar_text:
        products_info.append(extract_product_info(i))
   # print(products_info)
    return products_info


# Test json . The api is tested 
# {
#     "text": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Sample E-commerce Site</title>\n</head>\n<body>\n<header>\n    <div class=\"container\">\n        <h1>My E-commerce Site</h1>\n        <ul>\n            <li><a href=\"#\">Home</a></li>\n            <li><a href=\"#\">Products</a></li>\n            <li><a href=\"#\">About</a></li>\n            <li><a href=\"#\">Contact</a></li>\n        </ul>\n    </div>\n</header>\n<div class=\"container\">\n    <h2>Our Products</h2>\n    <div class=\"products\">\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Gel Pen\">\n            <h3>Gel Pen</h3>\n            <p class=\"price\">$2.99</p>\n            <p class=\"description\">Smooth writing gel pen with a comfortable grip and vibrant ink colors. Perfect for everyday use at school, home, or office.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Notebook\">\n            <h3>Notebook</h3>\n            <p class=\"price\">$5.99</p>\n            <p class=\"description\">Durable and stylish notebook with lined pages. Ideal for note-taking, journaling, or sketching. Available in various colors and sizes.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Wireless Mouse\">\n            <h3>Wireless Mouse</h3>\n            <p class=\"price\">$14.99</p>\n            <p class=\"description\">Ergonomic wireless mouse with high precision and long battery life. Compatible with Windows and MacOS. Perfect for work and gaming.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Bluetooth Speaker\">\n            <h3>Bluetooth Speaker</h3>\n            <p class=\"price\">$29.99</p>\n            <p class=\"description\">Portable Bluetooth speaker with excellent sound quality and deep bass. Water-resistant design makes it ideal for outdoor use.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Coffee Mug\">\n            <h3>Coffee Mug</h3>\n            <p class=\"price\">$9.99</p>\n            <p class=\"description\">Ceramic coffee mug with a comfortable handle and fun design. Microwave and dishwasher safe. Perfect for coffee, tea, or hot chocolate.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Smartphone Case\">\n            <h3>Smartphone Case</h3>\n            <p class=\"price\">$12.99</p>\n            <p class=\"description\">Durable and stylish smartphone case that provides excellent protection against scratches and drops. Available in various colors.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Water Bottle\">\n            <h3>Water Bottle</h3>\n            <p class=\"price\">$8.99</p>\n            <p class=\"description\">Reusable water bottle made from BPA-free materials. Features a leak-proof cap and is perfect for staying hydrated on the go.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Backpack\">\n            <h3>Backpack</h3>\n            <p class=\"price\">$49.99</p>\n            <p class=\"description\">Stylish and durable backpack with multiple compartments for organized storage. Ideal for school, work, or travel.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Desk Lamp\">\n            <h3>Desk Lamp</h3>\n            <p class=\"price\">$24.99</p>\n            <p class=\"description\">Adjustable desk lamp with LED light. Features multiple brightness levels and a modern design. Perfect for study or work.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Headphones\">\n            <h3>Headphones</h3>\n            <p class=\"price\">$59.99</p>\n            <p class=\"description\">Over-ear headphones with excellent sound quality and noise cancellation. Comfortable fit and long battery life. Ideal for music lovers.</p>\n        </div>\n        <div class=\"product\">\n            <img src=\"https://via.placeholder.com/150\" alt=\"Fitness Tracker\">\n            <h3>Fitness Tracker</h3>\n            <p class=\"price\">$39.99</p>\n            <p class=\"description\">Waterproof fitness tracker with heart rate monitor, step counter, and sleep tracking. Syncs with your smartphone for detailed insights.</p>\n        </div>\n    </div>\n</div>\n</body>\n</html>\n"
# }
##