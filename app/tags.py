import requests
from bs4 import BeautifulSoup

with open("sample.html","r") as f:
    html_doc = f.read()

soup = BeautifulSoup(html_doc, 'html.parser')

#print(soup.prettify())
#print(soup.title, type(soup.title))
#print(soup.title.name)
#print(soup.div)
#print(soup.find_all("div")[0])

# for link in soup.find_all("a"):
#     print(link.get("href"))
#     print(link.get_text())


# print(soup.select("div.italic"))

# print(soup.find(class_ = "italic"))

# for child in soup.find(class_ = "container").children:
#     print(child)


for parent in soup.find(class_ = "box").parent:
    print(parent)