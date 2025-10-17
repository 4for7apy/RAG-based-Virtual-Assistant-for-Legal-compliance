from bs4 import BeautifulSoup
import requests

# Define the URL to scrape
url = 'https://www.infosys.com/about.html'

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

print(soup.get_text(strip = True))