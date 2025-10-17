from bs4 import BeautifulSoup
import requests

def scrape_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text(strip=True)

def main():
    urls = []
    num_urls = int(input("Enter the number of URLs you want to scrape: "))
    
    # Prompt the user to enter each URL
    for i in range(num_urls):
        url = input(f"Enter URL {i+1}: ")
        urls.append(url)
    
    # Scraping each URL and saving the text to a file
    with open("scraped_all_text.txt", "w", encoding="utf-8") as file:
        for url in urls:
            text = scrape_url(url)
            file.write(f"URL: {url}\n")
            file.write(text + "\n\n")

    print("Scraping complete. The text has been saved to 'scraped_text.txt'.")

if __name__ == "__main__":
    main()
