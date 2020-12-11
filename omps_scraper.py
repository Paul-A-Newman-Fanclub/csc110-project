"""
CSC110 Final Project - Ozone Layer Visualization

Tobey Brizuela, Daniel Lazaro, Matthew Parvaneh, Michael Umeh
"""
import scrapy


base_url = 'https://ozonewatch.gsfc.nasa.gov/data/omps/'
directories = [base_url + 'Y' + str(year) for year in range(2012, 2021)]


class OzoneSpider(scrapy.Spider):
    """Our spider that scrapes quotes."""

    # the name of the spider.
    name = "ozone"

    def start_requests(self) -> None:
        """Creates the request objects for all the urls the spider encounters."""
        urls = directories

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        """
        A method that can be called to handle the response downloaded for
        each of the requests made.

        Response is of the object type TextResponse - hold's the page's content.

        This method usually parses the response, extracting the scraped data as
        dicts and also finding new URLs to follow and creating Requests (type Request)
        from them.
        """
        page = response.url.split("/")[-2]

        yield {str(page): response.css('a::attr(href)').getall()}
