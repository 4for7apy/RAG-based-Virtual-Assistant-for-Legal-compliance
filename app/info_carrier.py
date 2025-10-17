from bs4 import BeautifulSoup
import requests


url = 'https://www.infosys.com/careers/'

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

print(soup.get_text(strip = True))
