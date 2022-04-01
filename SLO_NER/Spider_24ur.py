import errno
import logging
import os
import random
from os import path
from time import sleep

import w3lib.html
import w3lib.html
from inline_requests import inline_requests
from scrapy import Spider, Request
from scrapyscript import Job, Processor

from helper_functions import write_pickle


class Spider_24ur(Spider):
    name = 'Spider_24ur'

    # enable fake-useragent python package
    custom_settings = {
        "DOWNLOADER_MIDDLEWARES": {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
            'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
            'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
            'scrapy_fake_useragent.middleware.RetryUserAgentMiddleware': 401,
        },
        'FAKEUSERAGENT_PROVIDERS': [
            'scrapy_fake_useragent.providers.FakeUserAgentProvider',  # this is the first provider we'll try
            'scrapy_fake_useragent.providers.FakerProvider',
            # if FakeUserAgentProvider fails, we'll use faker to generate a user-agent string for us
            'scrapy_fake_useragent.providers.FixedUserAgentProvider',  # fall back to USER_AGENT value
        ]
    }

    def __init__(self, page, url, **kw):
        self.page = page
        self.log = logging.getLogger('scrapy')
        self.log.setLevel(logging.WARNING)
        self.start_urls = [url]
        super().__init__(**kw)

    @inline_requests
    def parse(self, response):
        sleep(3)
        text = ""
        yield_out = {
            "continue": True
        }
        # GET URLS FOR ALL THE ARTICLES ON THE PAGE
        articles = response.xpath('//a[@class="timeline__item grid"]')
        if len(articles) != 20:
            yield_out["continue"] = False
        for article in articles:
            try:
                # Scrape article text, summary and date
                article_resp = yield Request("https://www.24ur.com" + article.attrib["href"])
                article_text = ([w3lib.html.remove_tags(x).strip('\xa0').strip("\'") for x in
                                 article_resp.xpath('//div[@class="article__body"]//span//p').extract() if
                                 x is not None])
                article_summary = ([w3lib.html.remove_tags(x).strip('\xa0').strip("\'") for x in
                                    article_resp.xpath('//div[@class="article__summary"]').extract() if
                                    x is not None])
                text = ' '.join(article_summary)
                text = text + ' '.join(article_text)
                article_date = ((article_resp.xpath('//p[@class="article__info"]/text()').get().split(","))[1]).split(
                    ".")
                # Remove space before date
                article_date[0] = article_date[0][1:]
                # Create dir structure to write into
                filename = './data/24ur/' + article_date[2] + '/' + ".".join(article_date[:2]) + \
                           '_'.join((article.attrib["href"].split('/'))).split('.')[0] + ".txt"
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                # Write extracted text to file
                try:
                    if not path.exists(filename):
                        with open(filename, 'w') as f:
                            f.write(text)
                    else:
                        break
                except:
                    self.log.error("Error saving file: ", filename)
                # Simple rate limiting
                sleep(random.uniform(0.5, 1))
            except:
                self.log.error("Failed request %s", article.attrib["href"], exc_info=True)

        # Save progress
        write_pickle(self.page, './progress/24ur.pickle')
        yield yield_out


if __name__ == '__main__':
    job = Job(Spider_24ur, 1, 'https://www.24ur.com/arhiv')
    processor = Processor(settings=None)
    processor.run([job])
    # try:
    #     data = processor.run([wemportalJob])[0]
    #     classla.download('sl')
    #     nlp = classla.Pipeline("sl", processors='tokenize,ner')
    #     doc = nlp(data["res"][0])
    #     print(doc.to_conll())
    # except IndexError:
    #     print("err")
