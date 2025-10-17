import requests
from bs4 import BeautifulSoup

# URLs to scrape
urls = [
    'https://www.sanfoundry.com/data-structure-questions-answers-rod-cutting/'
]
all_cleaned_text = ''

# Iterate through the URLs and scrape the text
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')  # Extract all paragraphs
    cleaned_paragraphs = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]  # Clean the text
    formatted_paragraphs = '\n\n'.join(cleaned_paragraphs)  # Join paragraphs with double newline for separation
    all_cleaned_text += formatted_paragraphs + '\n\n\n'  # Add triple newline as a separator between each URL's text

# Save the combined cleaned text to a text file
with open("mcq.pdf", "w", encoding="utf-8") as file:
    file.write(all_cleaned_text)

print("Data saved to mcq.txt successfully!")
