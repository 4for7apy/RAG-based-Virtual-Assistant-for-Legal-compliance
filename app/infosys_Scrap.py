import requests
from bs4 import BeautifulSoup

# Define the URL to scrape
url = 'https://www.infosys.com/'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the response
soup = BeautifulSoup(response.text, 'html.parser')

# Uncomment this block if you want to print the prettified HTML content
"""
soup = BeautifulSoup(response.content, 'html.parser')
print(soup.prettify())
"""

# Extract all the <a> tags and their titles
"""
for link in soup.find_all('a'):
    print(link.get('title'))
"""

# Uncomment this block if you want to print the title of the page

print(soup.get_text(strip = True))


# Write the titles of all <a> tags to a file
"""
with open('scraped_data.txt', 'w') as file:
    for link in soup.find_all('a'):
        file.write(link.get('title') + '\n')
"""
"""
# Get the text content of the page
text_content = soup.get_text()

# Write the text content to a file
with open('scraped_data1.txt', 'w') as file:
    file.write(text_content)
"""