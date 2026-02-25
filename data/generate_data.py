
import json
import random

# Lists to create variety
vendors = ["Amazon", "Walmart", "Starbucks", "Apple Store", "Best Buy", "Mario's Pizza", "Shell Gas", "Netflix", "Adobe", "IKEA", "Target", "Home Depot", "Uber", "Lyft", "Local Coffee", "City Parking"]
items = ["Laptop", "Coffee", "Office Chair", "Software Subscription", "Fuel", "Groceries", "Desk Lamp", "Monitor", "Headphones", "Dinner", "Pizza", "Books", "Printer Ink", "Smartphone", "Cables"]
currencies = ["USD", "EUR", "GBP", "CAD"]
date_formats = ["Oct 12, 2023", "12/10/2023", "yesterday", "last Friday", "2023-10-12", "Jan 5th", "05-01-2024"]

def generate_entry():
    vendor = random.choice(vendors)
    item = random.choice(items)
    qty = random.randint(1, 5)
    price = round(random.uniform(5.0, 2000.0), 2)
    currency = random.choice(currencies)
    date = random.choice(date_formats)
    
    # Randomize the sentence structure (The "Messy" Input)
    templates = [
        f"I bought {qty} {item} from {vendor} for {price} {currency} on {date}.",
        f"{date}: {vendor} purchase of {qty}x {item}. Total cost: {price} {currency}.",
        f"Spent {price} {currency} at {vendor} for {qty} {item} today.",
        f"Invoice from {vendor} for {qty} {item}. Amount: {price}. Date: {date}.",
        f"Just got {item} ({qty}) at {vendor}. It was {price} {currency} on {date}."
    ]
    
    input_text = random.choice(templates)
    
    # The expected structured output
    output_json = {
        "item": item,
        "quantity": qty,
        "date": date,
        "vendor": vendor,
        "total": price,
        "currency": currency
    }
    
    return {
        "instruction": "Extract invoice details into JSON.",
        "input": input_text,
        "output": json.dumps(output_json)
    }

# Generate 600 entries
dataset = [generate_entry() for _ in range(600)]

# Save to JSONL
with open("dataset.jsonl", "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

print(f"Successfully generated {len(dataset)} examples in dataset.jsonl")